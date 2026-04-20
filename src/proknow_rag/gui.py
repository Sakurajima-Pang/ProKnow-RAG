from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import gradio as gr

from proknow_rag.common.config import Settings
from proknow_rag.common.exceptions import ProKnowRAGError
from proknow_rag.common.gpu_monitor import get_gpu_memory_info, get_gpu_name
from proknow_rag.common.logging_config import setup_logging
from proknow_rag.data_preparation.manager import DataManager
from proknow_rag.index_construction.index_builder import IndexBuilder
from proknow_rag.index_construction.qdrant_store import QdrantEmbeddedStore
from proknow_rag.retrieval.compressor import ContextCompressor
from proknow_rag.retrieval.hybrid_search import HybridSearcher
from proknow_rag.retrieval.query_router import QueryRouter
from proknow_rag.retrieval.reranker import BGEReranker
from proknow_rag.retrieval.validators import validate_and_sanitize

logger = logging.getLogger(__name__)

_log_lines: list[str] = []

_cached_index_builder: IndexBuilder | None = None
_cached_searcher: HybridSearcher | None = None
_cached_reranker: BGEReranker | None = None


class LogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        _log_lines.append(line)
        if len(_log_lines) > 500:
            _log_lines.pop(0)


def _setup_log_capture() -> None:
    handler = LogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger("proknow_rag").addHandler(handler)
    logging.getLogger().addHandler(handler)


def _get_index_builder(settings: Settings) -> IndexBuilder:
    global _cached_index_builder
    if _cached_index_builder is None:
        logger.info("首次加载 IndexBuilder（含 BGE-M3 模型）...")
        _cached_index_builder = IndexBuilder(settings)
    return _cached_index_builder


def _get_search_components(settings: Settings) -> tuple[HybridSearcher, BGEReranker]:
    global _cached_searcher, _cached_reranker
    if _cached_searcher is None:
        logger.info("首次加载检索组件（含 BGE-M3 + Reranker 模型）...")
        _cached_searcher = HybridSearcher(settings)
        _cached_reranker = BGEReranker(settings=settings)
    return _cached_searcher, _cached_reranker


def get_system_info() -> str:
    settings = Settings()
    lines = ["## 🖥️ 系统信息", ""]

    gpu_info = get_gpu_memory_info()
    if gpu_info["total_mb"] > 0:
        used_pct = gpu_info["used_mb"] / gpu_info["total_mb"] * 100
        bar_len = 20
        filled = int(bar_len * used_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        gpu_name = get_gpu_name()
        lines.append("### GPU")
        lines.append(f"- **型号**: {gpu_name}")
        lines.append(f"- **总显存**: {gpu_info['total_mb']} MB")
        lines.append(f"- **已用**: {gpu_info['used_mb']} MB ({used_pct:.1f}%)")
        lines.append(f"- **可用**: {gpu_info['free_mb']} MB")
        lines.append(f"- `[{bar}] {used_pct:.1f}%`")
    else:
        lines.append("### GPU: ❌ 不可用")
    lines.append("")

    lines.append("### Qdrant 向量数据库")
    try:
        store = QdrantEmbeddedStore(settings)
        collections = store.client.get_collections().collections
        if not collections:
            lines.append("- ⚠️ 无 Collection（请先索引数据）")
        for coll in collections:
            try:
                info = store.get_collection_info(coll.name)
                lines.append(f"- **{coll.name}**: {info['points_count']} 点 | {info['vectors_count']} 索引向量 | {info['status']}")
            except Exception as e:
                lines.append(f"- **{coll.name}**: 获取信息失败 ({e})")
    except Exception as e:
        lines.append(f"- ❌ 连接失败: {e}")
    lines.append("")

    lines.append("### 模型状态")
    embedder_status = "✅ 已加载（内存中）" if _cached_index_builder is not None else "⏳ 未加载（首次索引时加载）"
    lines.append(f"- **BGE-M3 Embedder**: {embedder_status}")
    reranker_status = "✅ 已加载（内存中）" if _cached_reranker is not None else "⏳ 未加载（首次检索时加载）"
    lines.append(f"- **BGE-Reranker**: {reranker_status}")

    bge_m3_path = Path(settings.bge_m3_model_path)
    if bge_m3_path.exists():
        size_mb = sum(f.stat().st_size for f in bge_m3_path.rglob("*") if f.is_file()) / (1024 * 1024)
        lines.append(f"- **BGE-M3 磁盘**: ✅ ({size_mb:.0f} MB)")
    else:
        lines.append(f"- **BGE-M3 磁盘**: ❌ 不存在 (`{bge_m3_path}`)")

    reranker_path = Path(settings.bge_reranker_model_path)
    if reranker_path.exists():
        size_mb = sum(f.stat().st_size for f in reranker_path.rglob("*") if f.is_file()) / (1024 * 1024)
        lines.append(f"- **Reranker 磁盘**: ✅ ({size_mb:.0f} MB)")
    else:
        lines.append(f"- **Reranker 磁盘**: ❌ 不存在 (`{reranker_path}`)")
    lines.append("")

    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        file_count = sum(1 for f in data_dir.rglob("*") if f.is_file())
        lines.append(f"- **数据目录**: ✅ 存在 ({file_count} 文件)")
    else:
        lines.append(f"- **数据目录**: ⚠️ 不存在 (`{data_dir}`)")
    lines.append("")

    lines.append("### 配置")
    lines.append(f"- Qdrant 存储: `{settings.qdrant_storage_path}`")
    lines.append(f"- BGE-M3 路径: `{settings.bge_m3_model_path}`")
    lines.append(f"- Reranker 路径: `{settings.bge_reranker_model_path}`")
    lines.append(f"- 数据目录: `{settings.data_dir}`")
    lines.append(f"- HF 镜像: `{settings.hf_endpoint}`")

    return "\n".join(lines)


def do_index(directory: str, collection: str, batch_size: int) -> str:
    if not directory.strip():
        return "❌ 请输入目录路径"
    dir_path = Path(directory.strip())
    if not dir_path.is_absolute():
        settings = Settings()
        dir_path = Path(settings.data_dir) / directory.strip()
    if not dir_path.exists():
        return f"❌ 目录不存在: `{dir_path}`\n\n💡 请检查路径是否正确，或先创建目录并放入数据文件"
    if not dir_path.is_dir():
        return f"❌ 不是目录: `{dir_path}`"

    try:
        settings = Settings()
        data_manager = DataManager()
        index_builder = _get_index_builder(settings)

        start = time.perf_counter()
        chunks = data_manager.process_directory(str(dir_path))
        if not chunks:
            return "⚠️ 未发现可处理的文件"

        result = index_builder.build(chunks, collection_name=collection, batch_size=batch_size)
        elapsed = time.perf_counter() - start

        lines = [
            "## ✅ 索引构建完成", "",
            f"- **Collection**: {result.collection_name}",
            f"- **索引文档数**: {result.num_indexed}",
            f"- **失败文档数**: {len(result.failed_ids)}",
            f"- **索引版本**: {result.index_version}",
            f"- **耗时**: {elapsed:.2f}s",
        ]
        if result.failed_ids:
            lines.append(f"- **失败 ID**: {', '.join(result.failed_ids[:10])}")

        return "\n".join(lines)
    except ProKnowRAGError as e:
        return f"❌ 索引失败: {e}"
    except Exception as e:
        return f"❌ 未知错误: {e}\n\n```\n{traceback.format_exc()}\n```"


def do_search(query: str, collection: str, top_k: int, rerank_top_k: int) -> tuple[str, str]:
    if not query.strip():
        return "❌ 请输入查询内容", ""

    try:
        sanitized = validate_and_sanitize(query)
    except ProKnowRAGError as e:
        return f"❌ 查询验证失败: {e}", ""

    try:
        settings = Settings()
        router = QueryRouter()
        strategy = router.route(sanitized)
        weights = {
            "dense": strategy.dense_weight,
            "sparse": strategy.sparse_weight,
            "bm25": strategy.bm25_weight,
        }

        searcher, reranker = _get_search_components(settings)
        compressor = ContextCompressor()

        search_start = time.perf_counter()
        results = searcher.search(
            sanitized,
            collection_name=collection,
            limit=top_k,
            weights=weights,
            doc_type=strategy.doc_type_filter,
        )
        search_elapsed = time.perf_counter() - search_start

        if not results:
            meta = f"查询: `{sanitized}`\n路由: dense={weights['dense']:.2f} sparse={weights['sparse']:.2f}\n⚠️ 未找到相关结果"
            return meta, ""

        rerank_start = time.perf_counter()
        reranked = reranker.rerank(sanitized, results, top_k=rerank_top_k)
        rerank_elapsed = time.perf_counter() - rerank_start

        meta_lines = [
            f"查询: `{sanitized}`",
            f"路由: dense={weights['dense']:.2f} sparse={weights['sparse']:.2f} bm25={weights['bm25']:.2f}",
            f"检索: {search_elapsed*1000:.1f}ms | 重排: {rerank_elapsed*1000:.1f}ms | 结果: {len(reranked)} 条",
        ]
        meta = "\n".join(meta_lines)

        result_lines = []
        for i, doc in enumerate(reranked, 1):
            score = doc.score if hasattr(doc, "score") else 0.0
            source = doc.payload.get("source", "unknown") if hasattr(doc, "payload") and doc.payload else "unknown"
            doc_type = doc.payload.get("doc_type", "") if hasattr(doc, "payload") and doc.payload else ""
            content = doc.content if hasattr(doc, "content") else str(doc)
            compressed = compressor.compress(content, sanitized)

            result_lines.append(f"### [{i}] Score: {score:.4f}")
            if doc_type:
                result_lines.append(f"类型: `{doc_type}` | 来源: `{source}`")
            else:
                result_lines.append(f"来源: `{source}`")
            result_lines.append(f"\n> {compressed[:500]}{'...' if len(compressed) > 500 else ''}")
            result_lines.append("---")

        return meta, "\n".join(result_lines)
    except ProKnowRAGError as e:
        return f"❌ 检索失败: {e}", ""
    except Exception as e:
        return f"❌ 未知错误: {e}\n\n```\n{traceback.format_exc()}\n```", ""


def get_collection_stats() -> str:
    settings = Settings()
    try:
        store = QdrantEmbeddedStore(settings)
        collections = store.client.get_collections().collections
        if not collections:
            return "⚠️ 无 Collection"
        lines = []
        for coll in collections:
            try:
                info = store.get_collection_info(coll.name)
                lines.append(f"### {coll.name}")
                lines.append(f"- 向量数: {info['points_count']}")
                lines.append(f"- 索引数: {info['vectors_count']}")
                lines.append(f"- 状态: {info['status']}")
                lines.append("")
            except Exception as e:
                lines.append(f"### {coll.name}: 获取信息失败 ({e})")
                lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ 连接失败: {e}"


def delete_collection(collection_name: str) -> str:
    if not collection_name.strip():
        return "❌ 请输入 Collection 名称"
    settings = Settings()
    try:
        store = QdrantEmbeddedStore(settings)
        store.client.delete_collection(collection_name.strip())
        return f"✅ 已删除 Collection: {collection_name.strip()}"
    except Exception as e:
        return f"❌ 删除失败: {e}"


def get_logs() -> str:
    if not _log_lines:
        return "暂无日志"
    return "\n".join(_log_lines[-200:])


def build_gui() -> gr.Blocks:
    settings = Settings()
    default_data_dir = settings.data_dir

    with gr.Blocks(
        title="ProKnow-RAG 管理界面",
        theme=gr.themes.Soft(),
        css="""
        .contain { max-width: 1200px; margin: auto; }
        .log-box { font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; }
        """,
    ) as app:
        gr.Markdown("# 🔍 ProKnow-RAG 管理界面\n专业性知识检索系统 | Professional Knowledge Retrieval System")

        with gr.Tabs():
            with gr.Tab("📊 系统信息"):
                info_output = gr.Markdown(value=get_system_info(), elem_classes=["contain"])
                refresh_btn = gr.Button("🔄 刷新", variant="secondary")
                refresh_btn.click(fn=get_system_info, outputs=info_output)
                gr.Timer(value=30, active=True).tick(fn=get_system_info, outputs=info_output)

            with gr.Tab("📥 索引管理"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 构建索引")
                        index_dir = gr.Textbox(label="数据目录路径", placeholder="例如: D:\\docs 或绝对路径", value=default_data_dir)
                        with gr.Row():
                            index_collection = gr.Textbox(label="Collection 名称", value="proknow_rag")
                            index_batch = gr.Slider(label="批处理大小", minimum=2, maximum=32, value=12, step=2)
                        index_btn = gr.Button("🚀 开始索引", variant="primary")
                        index_output = gr.Markdown()

                    with gr.Column(scale=1):
                        gr.Markdown("### Collection 管理")
                        stats_output = gr.Markdown(value=get_collection_stats())
                        stats_btn = gr.Button("🔄 刷新统计", variant="secondary")
                        with gr.Row():
                            del_collection_name = gr.Textbox(label="Collection 名称", placeholder="要删除的 Collection")
                            del_btn = gr.Button("🗑️ 删除", variant="stop")
                        del_output = gr.Markdown()

                index_btn.click(fn=do_index, inputs=[index_dir, index_collection, index_batch], outputs=index_output)
                stats_btn.click(fn=get_collection_stats, outputs=stats_output)
                del_btn.click(fn=delete_collection, inputs=del_collection_name, outputs=del_output)

            with gr.Tab("🔎 检索测试"):
                with gr.Row():
                    with gr.Column(scale=3):
                        search_query = gr.Textbox(label="查询", placeholder="输入检索查询...", lines=2)
                        with gr.Row():
                            search_collection = gr.Textbox(label="Collection", value="proknow_rag")
                            search_top_k = gr.Slider(label="检索数量", minimum=5, maximum=50, value=20, step=5)
                            search_rerank_k = gr.Slider(label="重排数量", minimum=1, maximum=20, value=5, step=1)
                        search_btn = gr.Button("🔍 检索", variant="primary")
                        search_meta = gr.Markdown()
                    with gr.Column(scale=2):
                        search_results = gr.Markdown()

                search_btn.click(fn=do_search, inputs=[search_query, search_collection, search_top_k, search_rerank_k], outputs=[search_meta, search_results])

            with gr.Tab("📋 日志"):
                log_output = gr.Textbox(label="运行日志", value=get_logs(), lines=25, max_lines=50, elem_classes=["log-box"], interactive=False, show_copy_button=True)
                with gr.Row():
                    log_refresh_btn = gr.Button("🔄 刷新日志", variant="secondary")
                    log_clear_btn = gr.Button("🗑️ 清空日志")

                def clear_logs() -> str:
                    _log_lines.clear()
                    return "日志已清空"

                log_refresh_btn.click(fn=get_logs, outputs=log_output)
                log_clear_btn.click(fn=clear_logs, outputs=log_output)
                gr.Timer(value=5, active=True).tick(fn=get_logs, outputs=log_output)

        return app


def main() -> None:
    setup_logging()
    _setup_log_capture()

    app = build_gui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
