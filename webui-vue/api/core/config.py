from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# ============================================================================
# Paths & Constants
# ============================================================================

# Project structure
API_DIR = Path(__file__).parent.parent
WEBUI_DIR = API_DIR.parent
PROJECT_ROOT = WEBUI_DIR.parent

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

# 从 .env 读取路径配置 (支持相对路径)
def _resolve_path(env_key: str, default: str) -> Path:
    """解析路径，支持相对路径（相对于项目根目录）"""
    p = Path(os.getenv(env_key, default))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

MODEL_PATH = _resolve_path("MODEL_PATH", "./zimage_models")
LORA_PATH = _resolve_path("LORA_PATH", "./output")
DATASETS_DIR = _resolve_path("DATASET_PATH", "./datasets")

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_BASE_DIR = LORA_PATH  # LoRA 输出目录

# Create necessary directories
OUTPUTS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LORA_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Pydantic Models
# ============================================================================

class DatasetScanRequest(BaseModel):
    path: str

class CaptionRequest(BaseModel):
    path: str
    caption: Optional[str] = None

class CacheGenerateRequest(BaseModel):
    datasetPath: str
    generateLatent: bool = True
    generateText: bool = True
    vaePath: str = ""
    textEncoderPath: str = ""

class TrainingConfig(BaseModel):
    outputDir: str = "./output"
    outputName: str = "zimage-lora"
    modelPath: str = ""
    vaePath: str = ""
    textEncoderPath: str = ""
    datasetConfigPath: str = "./dataset_config.toml"
    cacheDir: str = "./cache"
    epochs: int = 10
    batchSize: int = 1
    learningRate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmupSteps: int = 100
    networkDim: int = 64
    networkAlpha: int = 64
    mixedPrecision: str = "bf16"
    gradientCheckpointing: bool = True
    gradientAccumulationSteps: int = 1
    maxGradNorm: float = 1.0
    seed: int = 42

class ConfigSaveRequest(BaseModel):
    path: str
    config: TrainingConfig

class SaveConfigRequest(BaseModel):
    name: str
    config: Dict[str, Any]

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 9
    guidance_scale: float = 1.0
    seed: int = -1
    width: int = 1024
    height: int = 1024
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    comparison_mode: bool = False

class DeleteHistoryRequest(BaseModel):
    timestamps: List[str]
