import logging

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem
            allocated = torch.cuda.memory_allocated(0)
            free = total - allocated
            return {"total_mb": total // 1024 // 1024, "allocated_mb": allocated // 1024 // 1024, "free_mb": free // 1024 // 1024}
    except ImportError:
        pass
    return {"total_mb": 0, "allocated_mb": 0, "free_mb": 0}


def check_gpu_available(min_mb: int = 2048) -> bool:
    info = get_gpu_memory_info()
    return info["free_mb"] >= min_mb
