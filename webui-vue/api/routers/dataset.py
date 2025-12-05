from fastapi import APIRouter, HTTPException, Form, UploadFile, File, BackgroundTasks
from pathlib import Path
from PIL import Image
import re
import shutil
import base64
import io
import requests
import asyncio
import threading
from typing import Optional
from pydantic import BaseModel

from core.config import DATASETS_DIR, DatasetScanRequest
from core import state

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

@router.post("/scan")
async def scan_dataset(request: DatasetScanRequest):
    """Scan a directory for images"""
    path = Path(request.path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {request.path}")
    
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.path}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    images = []
    total_size = 0
    
    # Recursively scan for images
    for file in path.rglob('*'):
        if file.is_file() and file.suffix.lower() in image_extensions:
            try:
                stat = file.stat()
                size = stat.st_size
                total_size += size
                
                # Get image dimensions
                with Image.open(file) as img:
                    width, height = img.size
                
                # Check for caption file
                caption = None
                caption_file = file.with_suffix('.txt')
                if caption_file.exists():
                    caption = caption_file.read_text(encoding='utf-8').strip()
                
                # Check for latent cache file (e.g. image_1024x1024_zi.safetensors)
                # We look for any file matching the pattern
                has_latent_cache = any(file.parent.glob(f"{file.stem}_*_zi.safetensors"))
                
                # Check for text encoder cache file
                text_cache = file.parent / f"{file.stem}_zi_te.safetensors"
                has_text_cache = text_cache.exists()
                
                images.append({
                    "path": str(file),
                    "filename": file.name,
                    "width": width,
                    "height": height,
                    "size": size,
                    "caption": caption,
                    "hasLatentCache": has_latent_cache,
                    "hasTextCache": has_text_cache,
                    "thumbnailUrl": f"/api/dataset/thumbnail?path={file}"
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
    
    return {
        "path": str(path),
        "name": path.name,
        "imageCount": len(images),
        "totalSize": total_size,
        "images": images
    }

@router.get("/list")
async def list_datasets():
    """List all datasets in the datasets folder"""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = []
    for folder in DATASETS_DIR.iterdir():
        if folder.is_dir():
            # Count images
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
            image_count = sum(1 for f in folder.rglob('*') if f.suffix.lower() in image_extensions)
            
            datasets.append({
                "name": folder.name,
                "path": str(folder),
                "imageCount": image_count
            })
    
    return {"datasets": datasets, "datasetsDir": str(DATASETS_DIR)}

@router.post("/create")
async def create_dataset(name: str = Form(...)):
    """Create a new dataset folder"""
    # Sanitize name
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name.strip())
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    
    dataset_path = DATASETS_DIR / safe_name
    if dataset_path.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    
    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(dataset_path), "name": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.post("/upload_batch")
async def upload_batch(
    dataset_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Upload multiple files to a dataset"""
    print(f"DEBUG: Received upload request for dataset '{dataset_name}' with {len(files)} files")
    
    # Sanitize dataset name
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', dataset_name.strip())
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
        
    dataset_path = DATASETS_DIR / safe_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Created/Checked dataset directory: {dataset_path}")
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        try:
            print(f"DEBUG: Processing file: {file.filename}, Type: {file.content_type}")
            # Skip non-image files (optional, but good practice)
            if not file.content_type.startswith('image/') and not file.filename.endswith('.txt') and not file.filename.endswith('.safetensors'):
                print(f"DEBUG: Skipping file {file.filename} due to type mismatch")
                continue
                
            file_path = dataset_path / file.filename
            
            # Ensure parent directory exists (for nested files)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"DEBUG: Saving to {file_path}")
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            uploaded_count += 1
        except Exception as e:
            print(f"ERROR: Failed to save {file.filename}: {e}")
            import traceback
            traceback.print_exc()
            errors.append(f"{file.filename}: {str(e)}")
            
    return {
        "success": True, 
        "uploaded": uploaded_count, 
        "dataset": safe_name,
        "errors": errors
    }

@router.post("/upload")
async def upload_files(
    dataset: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """Upload files to an existing dataset"""
    print(f"DEBUG: Upload to dataset '{dataset}' with {len(files)} files")
    
    # 查找数据集路径
    dataset_path = DATASETS_DIR / dataset
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found")
    
    uploaded = []
    errors = []
    
    for file in files:
        try:
            # 检查文件类型
            is_image = file.content_type and file.content_type.startswith('image/')
            is_txt = file.filename and file.filename.endswith('.txt')
            is_safetensors = file.filename and file.filename.endswith('.safetensors')
            
            if not (is_image or is_txt or is_safetensors):
                errors.append(f"{file.filename}: 不支持的文件类型")
                continue
            
            # 保存文件
            file_path = dataset_path / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded.append(file.filename)
            print(f"DEBUG: Saved {file.filename}")
            
        except Exception as e:
            print(f"ERROR: Failed to save {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")
    
    return {
        "success": True,
        "uploaded": uploaded,
        "datasetPath": str(dataset_path),
        "errors": errors
    }

@router.get("/thumbnail")
async def get_thumbnail(path: str):
    """Get a thumbnail of an image"""
    from fastapi.responses import Response
    import io
    
    img_path = Path(path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        with Image.open(img_path) as img:
            img.thumbnail((200, 200))
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            return Response(content=buffer.read(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image")
async def get_image(path: str):
    """Get full size image"""
    from fastapi.responses import FileResponse
    
    img_path = Path(path)
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    if not img_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
        
    return FileResponse(str(img_path))

@router.get("/cached")
async def list_cached_datasets():
    """List all datasets available for training (both raw and cached)"""
    try:
        datasets = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        
        if DATASETS_DIR.exists():
            # 递归扫描所有包含图片的目录
            def scan_directory(dir_path: Path, prefix: str = ""):
                for item in dir_path.iterdir():
                    if item.is_dir():
                        # 检查是否包含图片
                        image_count = sum(
                            1 for f in item.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions
                        )
                        
                        # 检查是否有缓存文件
                        cache_count = len(list(item.glob("*_zi.safetensors")))
                        
                        if image_count > 0 or cache_count > 0:
                            name = f"{prefix}{item.name}" if prefix else item.name
                            file_count = cache_count if cache_count > 0 else image_count
                            status = "已缓存" if cache_count > 0 else "图片"
                            
                            datasets.append({
                                "name": f"{name} ({file_count} {status})",
                                "path": str(item.absolute()),
                                "files": file_count,
                                "cached": cache_count > 0
                            })
                        
                        # 继续扫描子目录
                        scan_directory(item, f"{prefix}{item.name}/")
            
            scan_directory(DATASETS_DIR)
        
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@router.delete("/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset folder"""
    # Sanitize name to prevent directory traversal
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', dataset_name.strip())
    if not safe_name or safe_name == "." or safe_name == "..":
        raise HTTPException(status_code=400, detail="Invalid dataset name")
        
    dataset_path = DATASETS_DIR / safe_name
    
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        shutil.rmtree(dataset_path)
        return {"success": True, "message": f"Dataset '{dataset_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

class DeleteImagesRequest(BaseModel):
    paths: list[str]

@router.post("/delete-images")
async def delete_images(request: DeleteImagesRequest):
    """Delete specific images from a dataset"""
    deleted_count = 0
    errors = []
    
    for path_str in request.paths:
        try:
            path = Path(path_str)
            
            # Security check: ensure path is within DATASETS_DIR
            # Resolve to absolute path for comparison
            abs_path = path.resolve()
            try:
                abs_path.relative_to(DATASETS_DIR.resolve())
            except ValueError:
                errors.append(f"{path_str}: Access denied (outside datasets directory)")
                continue
                
            if abs_path.exists() and abs_path.is_file():
                abs_path.unlink()
                
                # Also try to delete associated files (.txt, .safetensors cache)
                txt_path = abs_path.with_suffix('.txt')
                if txt_path.exists():
                    txt_path.unlink()
                    
                latent_cache = abs_path.parent / f"{abs_path.stem}_zi.safetensors"
                if latent_cache.exists():
                    latent_cache.unlink()
                    
                text_cache = abs_path.parent / f"{abs_path.stem}_zi_te.safetensors"
                if text_cache.exists():
                    text_cache.unlink()
                    
                deleted_count += 1
            else:
                errors.append(f"{path_str}: File not found")
                
        except Exception as e:
            errors.append(f"{path_str}: {str(e)}")
            
    return {
        "success": True,
        "deleted": deleted_count,
        "errors": errors
    }


# ============================================================================
# Ollama 标注功能
# ============================================================================

class OllamaTagRequest(BaseModel):
    dataset_path: str
    ollama_url: str = "http://localhost:11434"
    model: str = "llava"
    prompt: str = "Describe this image in detail."
    max_long_edge: int = 1024  # 长边最大尺寸
    skip_existing: bool = True  # 跳过已有标注
    trigger_word: str = ""  # 触发词，添加到每个标注开头
    enable_think: bool = False  # 是否启用思考模式（某些模型如 deepseek、qwen3 支持）

# 标注状态
tagging_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": "",
    "errors": []
}

@router.get("/ollama/models")
async def get_ollama_models(ollama_url: str = "http://localhost:11434"):
    """获取 Ollama 可用模型列表"""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models, "success": True}
    except requests.exceptions.ConnectionError:
        return {"models": [], "success": False, "error": "无法连接到 Ollama 服务"}
    except Exception as e:
        return {"models": [], "success": False, "error": str(e)}

@router.get("/ollama/status")
async def get_tagging_status():
    """获取标注进度"""
    return tagging_state

@router.post("/ollama/stop")
async def stop_tagging():
    """停止标注"""
    tagging_state["running"] = False
    return {"success": True}

def resize_image_for_api(img_path: Path, max_long_edge: int) -> bytes:
    """缩放图片长边到指定尺寸，返回 JPEG 字节（参照用户脚本使用 getvalue）"""
    with Image.open(img_path) as img:
        # 打印原始图片信息用于调试
        print(f"[Ollama] Opening image: {img_path}")
        print(f"[Ollama] Original size: {img.size}, mode: {img.mode}")
        
        # 转换为 RGB
        if img.mode in ('RGBA', 'P', 'LA', 'L'):
            img = img.convert('RGB')
        
        w, h = img.size
        long_edge = max(w, h)
        
        if long_edge > max_long_edge:
            ratio = max_long_edge / long_edge
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.BICUBIC)
            print(f"[Ollama] Resized to: {new_size}")
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        # 使用 getvalue() 而非 read()，确保获取完整数据
        jpeg_bytes = buf.getvalue()
        print(f"[Ollama] JPEG bytes length: {len(jpeg_bytes)}")
        return jpeg_bytes

def generate_caption_ollama_sync(img_path: Path, ollama_url: str, model: str, prompt: str, max_long_edge: int, enable_think: bool = False) -> Optional[str]:
    """同步调用 Ollama API - 每次独立调用，不保留上下文
    
    Args:
        enable_think: 是否启用思考模式（某些模型如 deepseek、qwen3 支持）
                     如果模型不支持思考模式，建议设为 False
    """
    try:
        jpeg_bytes = resize_image_for_api(img_path, max_long_edge)
        base64_img = base64.b64encode(jpeg_bytes).decode()
        
        print(f"[Ollama] Processing: {img_path.name}")
        print(f"[Ollama] Image bytes: {len(jpeg_bytes)}, Base64 len: {len(base64_img)}")
        print(f"[Ollama] Model: {model}, URL: {ollama_url}, Think: {enable_think}")
        
        # 关键：context=[] 清除对话历史，确保每张图片独立处理
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64_img],
            "stream": False,
            "context": [],  # 清除上下文，避免累积图片
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # 添加思考模式控制（某些模型如 deepseek、qwen3 支持）
        # 注意：不是所有模型都支持这个参数，不支持的模型会忽略它
        if not enable_think:
            # 关闭思考模式，直接输出结果
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = 1024  # 限制输出长度
        
        print(f"[Ollama] Sending request to {ollama_url}/api/generate")
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
        print(f"[Ollama] Response status: {resp.status_code}")
        resp.raise_for_status()
        
        data = resp.json()
        print(f"[Ollama] Response keys: {data.keys()}")
        
        if "response" in data:
            caption = data["response"].strip()
            # 清理输出
            caption = caption.replace("```markdown", "").replace("```", "")
            import re
            # 移除思考标签内容（<think>...</think> 或 <thinking>...</thinking>）
            caption = re.sub(r'<think>.*?</think>', '', caption, flags=re.DOTALL)
            caption = re.sub(r'<thinking>.*?</thinking>', '', caption, flags=re.DOTALL)
            # 清理可能的 [img-X] 残留标记
            caption = re.sub(r'\[img-\d+\]', '', caption)
            # 清理多余的空行
            caption = re.sub(r'\n{3,}', '\n\n', caption)
            caption = caption.strip()
            
            if not caption:
                print(f"[Ollama] Warning: Empty caption after cleaning for {img_path.name}")
                return None
            
            print(f"[Ollama] Caption: {caption[:100]}...")
            return caption
        else:
            print(f"[Ollama] No 'response' in data: {data}")
        return None
    except requests.exceptions.Timeout:
        print(f"[Ollama] Timeout for {img_path.name} - try a smaller image or faster model")
        return None
    except Exception as e:
        print(f"[Ollama] Error for {img_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


class ResizeImagesRequest(BaseModel):
    dataset_path: str
    max_long_edge: int = 1024
    quality: int = 95  # JPEG 质量
    sharpen: float = 0.3  # 锐化强度 0-1

# 尝试导入 OpenCV
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def resize_with_pil_hq(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """使用 PIL 高质量多步下采样"""
    from PIL import ImageEnhance
    
    w, h = img.size
    
    # 计算目标尺寸
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    # 确保尺寸是偶数
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    
    # 多步下采样：每次最多缩小到70%
    current_img = img
    current_w, current_h = w, h
    
    while current_w > new_w * 1.5 or current_h > new_h * 1.5:
        step_w = int(current_w * 0.7)
        step_h = int(current_h * 0.7)
        step_w = max(step_w, new_w)
        step_h = max(step_h, new_h)
        current_img = current_img.resize((step_w, step_h), Image.LANCZOS)
        current_w, current_h = step_w, step_h
    
    # 最终缩放
    result = current_img.resize((new_w, new_h), Image.LANCZOS)
    
    # 锐化恢复细节
    if sharpen > 0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(1.0 + sharpen)
    
    return result

def resize_with_cv2_hq(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """使用 OpenCV INTER_AREA 高质量下采样 + USM锐化"""
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img_array.shape[:2]
    
    # 计算目标尺寸
    if w >= h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1
    
    # INTER_AREA 最佳抗锯齿下采样
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # USM 锐化
    if sharpen > 0:
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        sharpened = cv2.addWeighted(resized, 1.0 + sharpen, blurred, -sharpen, 0)
        resized = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(resized)

def resize_high_quality(img: Image.Image, target_size: int, sharpen: float = 0.3) -> Image.Image:
    """高质量图像缩放，优先使用 OpenCV"""
    if HAS_CV2:
        return resize_with_cv2_hq(img, target_size, sharpen)
    return resize_with_pil_hq(img, target_size, sharpen)

# 缩放状态
resize_state = {
    "running": False,
    "total": 0,
    "completed": 0,
    "current_file": ""
}

@router.get("/resize/status")
async def get_resize_status():
    """获取缩放进度"""
    return resize_state

@router.post("/resize/stop")
async def stop_resize():
    """停止缩放"""
    resize_state["running"] = False
    return {"success": True}

@router.post("/resize")
async def resize_images(request: ResizeImagesRequest):
    """批量缩放图片（按长边，不可撤销）"""
    global resize_state
    
    if resize_state["running"]:
        raise HTTPException(status_code=400, detail="缩放任务正在进行中")
    
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    # 收集图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        for img_path in dataset_path.rglob(f"*{ext}"):
            image_paths.append(img_path)
    
    if not image_paths:
        return {"success": True, "message": "没有图片需要处理", "total": 0}
    
    # 初始化状态
    resize_state = {
        "running": True,
        "total": len(image_paths),
        "completed": 0,
        "current_file": ""
    }
    
    # 后台执行缩放
    async def run_resize():
        global resize_state
        for img_path in image_paths:
            if not resize_state["running"]:
                break
            
            resize_state["current_file"] = img_path.name
            
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    long_edge = max(w, h)
                    
                    # 只处理超过目标尺寸的图片
                    if long_edge > request.max_long_edge:
                        # 转换为 RGB
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # 使用高质量缩放
                        img_resized = resize_high_quality(
                            img, 
                            request.max_long_edge, 
                            sharpen=request.sharpen
                        )
                        
                        # 保存（覆盖原文件）
                        if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                            img_resized.save(img_path, format='JPEG', quality=request.quality, subsampling=0)
                        elif img_path.suffix.lower() == '.png':
                            img_resized.save(img_path, format='PNG', compress_level=1)
                        elif img_path.suffix.lower() == '.webp':
                            img_resized.save(img_path, format='WEBP', quality=request.quality)
            except:
                pass  # 跳过无法处理的文件
            
            resize_state["completed"] += 1
            await asyncio.sleep(0.01)
        
        resize_state["running"] = False
        resize_state["current_file"] = ""
    
    asyncio.create_task(run_resize())
    
    return {
        "success": True,
        "message": f"开始处理 {len(image_paths)} 张图片",
        "total": len(image_paths)
    }

class DeleteCaptionsRequest(BaseModel):
    dataset_path: str

@router.post("/delete-captions")
async def delete_captions(request: DeleteCaptionsRequest):
    """删除数据集中所有 .txt 标注文件"""
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    deleted_count = 0
    errors = []
    
    for txt_file in dataset_path.rglob("*.txt"):
        try:
            txt_file.unlink()
            deleted_count += 1
        except Exception as e:
            errors.append(f"{txt_file.name}: {str(e)}")
    
    return {
        "success": True,
        "deleted": deleted_count,
        "errors": errors
    }

def run_tagging_thread(image_paths: list, ollama_url: str, model: str, prompt: str, max_long_edge: int, trigger_word: str = "", enable_think: bool = False):
    """在独立线程中执行标注任务"""
    global tagging_state
    
    for img_path in image_paths:
        if not tagging_state["running"]:
            break
        
        tagging_state["current_file"] = img_path.name
        
        caption = generate_caption_ollama_sync(
            img_path,
            ollama_url,
            model,
            prompt,
            max_long_edge,
            enable_think
        )
        
        if caption:
            # 如果有触发词，添加到标注开头
            if trigger_word.strip():
                caption = f"{trigger_word.strip()}, {caption}"
            txt_path = img_path.with_suffix('.txt')
            txt_path.write_text(caption, encoding='utf-8')
        else:
            tagging_state["errors"].append(img_path.name)
        
        tagging_state["completed"] += 1
    
    tagging_state["running"] = False
    tagging_state["current_file"] = ""

@router.post("/ollama/tag")
async def start_tagging(request: OllamaTagRequest):
    """开始批量标注"""
    global tagging_state
    
    if tagging_state["running"]:
        raise HTTPException(status_code=400, detail="标注任务正在进行中")
    
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    
    # 收集待标注图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []
    
    for ext in image_extensions:
        for img_path in dataset_path.rglob(f"*{ext}"):
            txt_path = img_path.with_suffix('.txt')
            if request.skip_existing and txt_path.exists():
                continue
            image_paths.append(img_path)
    
    if not image_paths:
        return {"success": True, "message": "没有需要标注的图片", "total": 0}
    
    # 初始化状态
    tagging_state = {
        "running": True,
        "total": len(image_paths),
        "completed": 0,
        "current_file": "",
        "errors": []
    }
    
    # 启动独立后台线程（不阻塞事件循环）
    thread = threading.Thread(
        target=run_tagging_thread,
        args=(image_paths, request.ollama_url, request.model, request.prompt, request.max_long_edge, request.trigger_word, request.enable_think),
        daemon=True
    )
    thread.start()
    
    return {
        "success": True,
        "message": f"开始标注 {len(image_paths)} 张图片",
        "total": len(image_paths)
    }
