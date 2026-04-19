from proknow_rag.common.config import Settings


def main():
    settings = Settings()
    print(f"ProKnow-RAG v{__import__('proknow_rag').__version__}")
    print(f"Qdrant storage: {settings.qdrant_storage_path}")
    print(f"BGE-M3 model: {settings.bge_m3_model_path}")


if __name__ == "__main__":
    main()
