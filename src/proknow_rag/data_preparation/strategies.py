from pathlib import Path
from typing import Any

import yaml

from proknow_rag.data_preparation.parsers.markdown_parser import MarkdownParser
from proknow_rag.data_preparation.parsers.pdf_parser import PdfParser
from proknow_rag.data_preparation.parsers.code_parser import CodeParser
from proknow_rag.data_preparation.parsers.paper_parser import PaperParser
from proknow_rag.data_preparation.parsers.llm_parser import LlmParser
from proknow_rag.data_preparation.chunkers.recursive_chunker import RecursiveChunker
from proknow_rag.data_preparation.chunkers.structure_chunker import StructureChunker
from proknow_rag.data_preparation.chunkers.ast_chunker import AstChunker
from proknow_rag.data_preparation.chunkers.semantic_chunker import SemanticChunker
from proknow_rag.data_preparation.chunkers.llm_chunker import LlmChunker
from proknow_rag.common.exceptions import DataPreparationError


_PARSER_REGISTRY = {
    "markdown_parser": MarkdownParser,
    "pdf_parser": PdfParser,
    "code_parser": CodeParser,
    "paper_parser": PaperParser,
    "llm_parser": LlmParser,
}

_CHUNKER_REGISTRY = {
    "recursive_chunker": RecursiveChunker,
    "structure_chunker": StructureChunker,
    "ast_chunker": AstChunker,
    "semantic_chunker": SemanticChunker,
    "llm_chunker": LlmChunker,
}

_DEFAULT_STRATEGIES: dict[str, dict[str, Any]] = {
    ".md": {
        "parser": "markdown_parser",
        "chunker": "structure_chunker",
        "config": {"overlap": 50},
    },
    ".pdf": {
        "parser": "pdf_parser",
        "chunker": "recursive_chunker",
        "config": {"chunk_size": 500, "overlap": 100},
    },
    ".py": {
        "parser": "code_parser",
        "chunker": "ast_chunker",
        "config": {},
    },
    ".tex": {
        "parser": "paper_parser",
        "chunker": "structure_chunker",
        "config": {"overlap": 100},
    },
}


class StrategyRegistry:
    def __init__(self):
        self._strategies: dict[str, dict[str, Any]] = dict(_DEFAULT_STRATEGIES)

    def get_strategy(self, extension: str) -> dict[str, Any]:
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in self._strategies:
            raise DataPreparationError(f"未注册的文件扩展名策略: {ext}")
        return self._strategies[ext]

    def register_strategy(self, extension: str, parser_name: str, chunker_name: str, config: dict | None = None):
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        self._strategies[ext] = {
            "parser": parser_name,
            "chunker": chunker_name,
            "config": config or {},
        }

    def load_from_yaml(self, yaml_path: str):
        path = Path(yaml_path)
        if not path.exists():
            raise DataPreparationError(f"策略配置文件不存在: {yaml_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise DataPreparationError(f"加载YAML配置失败: {yaml_path}, {e}")

        if not isinstance(config, dict):
            raise DataPreparationError(f"YAML配置格式错误: {yaml_path}")

        strategies = config.get("strategies", config)
        if not isinstance(strategies, dict):
            raise DataPreparationError(f"策略配置格式错误: {yaml_path}")

        for ext, strategy in strategies.items():
            parser_name = strategy.get("parser")
            chunker_name = strategy.get("chunker")
            strategy_config = strategy.get("config", {})

            if not parser_name or not chunker_name:
                raise DataPreparationError(f"策略缺少parser或chunker: {ext}")

            if parser_name not in _PARSER_REGISTRY:
                raise DataPreparationError(f"未注册的解析器: {parser_name}")
            if chunker_name not in _CHUNKER_REGISTRY:
                raise DataPreparationError(f"未注册的分块器: {chunker_name}")

            self.register_strategy(ext, parser_name, chunker_name, strategy_config)

    def create_parser(self, extension: str):
        strategy = self.get_strategy(extension)
        parser_name = strategy["parser"]
        parser_cls = _PARSER_REGISTRY[parser_name]
        return parser_cls()

    def create_chunker(self, extension: str):
        strategy = self.get_strategy(extension)
        chunker_name = strategy["chunker"]
        config = strategy.get("config", {})
        chunker_cls = _CHUNKER_REGISTRY[chunker_name]
        return chunker_cls(**config)

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._strategies.keys())

    @property
    def strategies(self) -> dict[str, dict[str, Any]]:
        return dict(self._strategies)
