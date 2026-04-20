import logging

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "total_mb": info.total // 1024 // 1024,
            "used_mb": info.used // 1024 // 1024,
            "free_mb": info.free // 1024 // 1024,
            "allocated_mb": info.used // 1024 // 1024,
        }
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            free = total - allocated
            return {
                "total_mb": total // 1024 // 1024,
                "used_mb": allocated // 1024 // 1024,
                "allocated_mb": allocated // 1024 // 1024,
                "free_mb": free // 1024 // 1024,
            }
    except ImportError:
        pass

    return {"total_mb": 0, "used_mb": 0, "allocated_mb": 0, "free_mb": 0}


def get_gpu_name() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return name
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return "Unknown"


def check_gpu_available(min_mb: int = 2048) -> bool:
    info = get_gpu_memory_info()
    return info["free_mb"] >= min_mb
