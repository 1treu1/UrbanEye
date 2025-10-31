"""GPU and TensorFlow setup for DeepFace."""

import os
import sistema_vigilancia.src.config as config


def ensure_tf_gpu() -> None:
    """Configure TensorFlow GPU settings for optimal performance.
    
    Sets memory growth to avoid pre-allocating full VRAM.
    This is a no-op on subsequent calls.
    """
    if config.GPU_READY:
        return
    
    # Prefer graceful memory growth to avoid pre-allocating full VRAM
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)  # type: ignore
            except Exception:
                pass
        # Optional XLA; keep conservative to avoid surprises
        try:
            tf.config.optimizer.set_jit(False)  # type: ignore
        except Exception:
            pass
    except Exception:
        # If TF not available, proceed silently (DeepFace may still use PyTorch)
        pass
    
    config.GPU_READY = True

