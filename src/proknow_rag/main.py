from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import structlog

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import ProKnowRAGError
from proknow_rag.common.gpu_monitor import get_gpu_memory_info, get_gpu_name
from proknow_rag.common.logging_config import setup_logging
from proknow_rag.data_preparation.manager import DataManager
from proknow_rag.index_construction.embedder import BGEM3Embedder
from proknow_rag.index_construction.index_builder import IndexBuilder
from proknow_rag.index_construction.qdrant_store import QdrantEmbeddedStore
from proknow_rag.retrieval.compressor import ContextCompressor
from proknow_rag.retrieval.hybrid_search import HybridSearcher
from proknow_rag.retrieval.query_router import QueryRouter
from proknow_rag.retrieval.reranker import BGEReranker
from proknow_rag.retrieval.validators import validate_and_sanitize

logger = structlog.get_logger(__name__)


def cmd_index(args: argparse.Namespace) -> None:
    settings = Settings()
    data_manager = DataManager()
    index_builder = IndexBuilder(settings)

    logger.info("开始索引构建", dir_path=args.directory)
    start = time.perf_counter()

    chunks = data_manager.process_directory(args.directory)
    if not chunks:
        print("未发现可处理的文件")
        return

    result = index_builder.build(
        chunks,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )

    elapsed = time.perf_counter() - start
    print(f"\n索引构建完成")
    print(f"  Collection:  {result.collection_name}")
    print(f"  索引文档数:  {result.num_indexed}")
    print(f"  失败文档数:  {len(result.failed_ids)}")
    print(f"  索引版本:    {result.index_version}")
    print(f"  耗时:        {elapsed:.2f}s")


def cmd_search(args: argparse.Namespace) -> None:
    settings = Settings()
    query = args.query

    try:
        query = validate_and_sanitize(query)
    except ProKnowRAGError as e:
        print(f"查询验证失败: {e}")
        return

    router = QueryRouter()
    strategy = router.route(query)
    weights = {
        "dense": strategy.dense_weight,
        "sparse": strategy.sparse_weight,
        "colbert": strategy.colbert_weight,
        "bm25": strategy.bm25_weight,
    }

    searcher = HybridSearcher(settings)
    reranker = BGEReranker(settings=settings)
    compressor = ContextCompressor()

    print(f"\n查询: {query}")
    print(f"路由策略: dense={weights['dense']:.2f} sparse={weights['sparse']:.2f} colbert={weights['colbert']:.2f}")

    search_start = time.perf_counter()
    results = searcher.search(
        query,
        collection_name=args.collection,
        limit=args.top_k,
        weights=weights,
        doc_type=strategy.doc_type_filter,
    )
    search_elapsed = time.perf_counter() - search_start

    if not results:
        print("未找到相关结果")
        return

    rerank_start = time.perf_counter()
    reranked = reranker.rerank(query, results, top_k=args.rerank_top_k)
    rerank_elapsed = time.perf_counter() - rerank_start

    compressed = []
    for doc in reranked:
        content = doc.content if hasattr(doc, "content") else str(doc)
        compressed_content = compressor.compress(content, query)
        compressed.append((doc, compressed_content))

    print(f"\n检索耗时: {search_elapsed * 1000:.2f}ms | 重排序耗时: {rerank_elapsed * 1000:.2f}ms")
    print(f"结果数: {len(compressed)}")
    print("-" * 60)

    for i, (doc, content) in enumerate(compressed, 1):
        score = doc.score if hasattr(doc, "score") else 0.0
        source = doc.payload.get("source", "unknown") if hasattr(doc, "payload") and doc.payload else "unknown"
        print(f"\n[{i}] Score: {score:.4f} | Source: {source}")
        print(f"  {content[:300]}{'...' if len(content) > 300 else ''}")


def cmd_info(args: argparse.Namespace) -> None:
    settings = Settings()

    print("=" * 50)
    print("ProKnow-RAG System Info")
    print("=" * 50)

    print("\n[GPU]")
    gpu_info = get_gpu_memory_info()
    if gpu_info["total_mb"] > 0:
        gpu_name = get_gpu_name()
        print(f"  Name:      {gpu_name}")
        print(f"  Total:     {gpu_info['total_mb']} MB")
        print(f"  Used:      {gpu_info['used_mb']} MB")
        print(f"  Free:      {gpu_info['free_mb']} MB")
    else:
        print("  GPU 不可用")

    print("\n[Qdrant]")
    try:
        store = QdrantEmbeddedStore(settings)
        collections = store.client.get_collections().collections
        if not collections:
            print("  无 Collection")
        for coll in collections:
            try:
                info = store.get_collection_info(coll.name)
                print(f"  Collection: {info['name']}")
                print(f"    Points:  {info['points_count']}")
                print(f"    Vectors: {info['vectors_count']}")
                print(f"    Status:  {info['status']}")
            except Exception as e:
                print(f"  Collection: {coll.name} (获取信息失败: {e})")
    except Exception as e:
        print(f"  Qdrant 连接失败: {e}")

    print("\n[Models]")
    try:
        embedder = BGEM3Embedder(settings)
        print(f"  BGE-M3 Embedder: 已加载 (GPU: {embedder.use_gpu})")
        del embedder
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  BGE-M3 Embedder: 未加载 ({e})")

    reranker_model_path = Path(settings.bge_reranker_model_path)
    if reranker_model_path.exists():
        print(f"  BGE Reranker:    模型文件存在 ({reranker_model_path})")
    else:
        print(f"  BGE Reranker:    模型文件不存在 (请运行 scripts/download_models.py)")

    print("\n[Settings]")
    print(f"  Qdrant Storage: {settings.qdrant_storage_path}")
    print(f"  BGE-M3 Model:   {settings.bge_m3_model_path}")
    print(f"  BGE Reranker:   {settings.bge_reranker_model_path}")
    print(f"  Data Dir:       {settings.data_dir}")
    print(f"  HF Endpoint:    {settings.hf_endpoint}")

    print("\n" + "=" * 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proknow-rag",
        description="ProKnow-RAG: Professional Knowledge Retrieval System",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    index_parser = subparsers.add_parser("index", help="索引指定目录的文件")
    index_parser.add_argument("directory", type=str, help="要索引的目录路径")
    index_parser.add_argument("--collection", type=str, default="proknow_rag", help="Collection 名称")
    index_parser.add_argument("--batch-size", type=int, default=12, help="嵌入批处理大小")

    search_parser = subparsers.add_parser("search", help="执行检索")
    search_parser.add_argument("query", type=str, help="查询文本")
    search_parser.add_argument("--collection", type=str, default="proknow_rag", help="Collection 名称")
    search_parser.add_argument("--top-k", type=int, default=20, help="检索返回数量")
    search_parser.add_argument("--rerank-top-k", type=int, default=5, help="重排序返回数量")

    subparsers.add_parser("info", help="显示系统信息")

    gui_parser = subparsers.add_parser("gui", help="启动 GUI 管理界面")
    gui_parser.add_argument("--port", type=int, default=7860, help="服务端口")
    gui_parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")

    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "index":
            cmd_index(args)
        elif args.command == "search":
            cmd_search(args)
        elif args.command == "info":
            cmd_info(args)
        elif args.command == "gui":
            from proknow_rag.gui import build_gui, _setup_log_capture
            setup_logging()
            _setup_log_capture()
            app = build_gui()
            app.launch(
                server_name="127.0.0.1",
                server_port=args.port,
                share=False,
                inbrowser=not args.no_browser,
            )
    except ProKnowRAGError as e:
        logger.error("执行失败", command=args.command, error=str(e))
        print(f"\n错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已中断")
        sys.exit(130)
    except Exception as e:
        logger.error("未知错误", command=args.command, error=str(e))
        print(f"\n未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
