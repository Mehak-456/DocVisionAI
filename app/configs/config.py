import os
import logging
from pydantic import BaseModel

logger = logging.getLogger("DocVisionAI.Config")

class Settings(BaseModel):
    # API Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    temp_dir: str = os.path.join(base_dir, "temp_uploads")
    static_dir: str = os.path.join(base_dir, "app", "static")

    # VLM Model Settings
    model_path: str = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
    adapter_path: str = os.getenv("ADAPTER_PATH", os.path.join(base_dir, "lora_adapters"))
    device: str = os.getenv("DEVICE", "auto")
    use_simulator: bool = os.getenv("USE_SIMULATOR", "True").lower() in ("true", "1", "yes")

    # MLOps Settings
    wandb_project: str = os.getenv("WANDB_PROJECT", "docvisionai")
    wandb_disabled: bool = os.getenv("WANDB_DISABLED", "True").lower() in ("true", "1", "yes")




# Instantiate settings
settings = Settings()

# Ensure directories exist
os.makedirs(settings.temp_dir, exist_ok=True)
os.makedirs(settings.static_dir, exist_ok=True)

logger.info(f"Loaded config: Simulator={settings.use_simulator}, Device={settings.device}, Host={settings.host}:{settings.port}")
