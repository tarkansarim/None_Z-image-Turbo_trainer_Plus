from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, List
from datetime import datetime
import json
import io
import base64
import torch
from pathlib import Path
from PIL import Image
import os

from core.config import OUTPUTS_DIR, PROJECT_ROOT, GenerationRequest, DeleteHistoryRequest, LORA_PATH
from core import state
from routers.websocket import sync_broadcast_generation_progress

# Compatibility shim for older diffusers versions with newer huggingface_hub
import huggingface_hub
try:
    from huggingface_hub import cached_download
except ImportError:
    # hf_hub_download is the new replacement, but cached_download signature might differ slightly
    # usually this is enough for simple cases
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download


router = APIRouter(tags=["generation"])

@router.get("/api/loras")
async def get_loras():
    """Scan for LoRA models in LORA_PATH directory"""
    loras = []
    
    if not LORA_PATH.exists():
        return []
        
    for root, _, files in os.walk(LORA_PATH):
        for file in files:
            if file.endswith(".safetensors"):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(LORA_PATH)
                loras.append({
                    "name": str(rel_path),
                    "path": str(full_path),
                    "size": full_path.stat().st_size
                })
    
    return sorted(loras, key=lambda x: x["name"])

@router.get("/api/loras/download")
async def download_lora(path: str):
    """Download a LoRA model file"""
    file_path = Path(path)
    
    # 安全检查：确保文件在 LORA_PATH 内
    try:
        file_path.resolve().relative_to(LORA_PATH.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.suffix == ".safetensors":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream"
    )

@router.delete("/api/loras/delete")
async def delete_lora(path: str):
    """Delete a LoRA model file"""
    file_path = Path(path)
    
    # 安全检查：确保文件在 LORA_PATH 内
    try:
        file_path.resolve().relative_to(LORA_PATH.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.suffix == ".safetensors":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        file_path.unlink()
        return {"success": True, "message": f"Deleted {file_path.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@router.post("/api/generate-stream")
async def generate_image_stream(req: GenerationRequest):
    """Generate image with streaming progress updates (SSE)"""
    from fastapi.responses import StreamingResponse
    import asyncio
    import queue
    import threading
    import time
    
    progress_queue = queue.Queue()
    result_holder = {"result": None, "error": None}
    
    def do_generate_with_queue():
        """在线程中执行生成，通过队列发送进度"""
        from diffusers import ZImagePipeline
        
        try:
            progress_queue.put({"stage": "loading", "step": 0, "total": req.steps, "message": "Loading model..."})
            
            # Load model if not loaded
            if state.pipeline is None:
                from core.config import MODEL_PATH
                if not MODEL_PATH.exists():
                    raise Exception("Model not found")
                state.pipeline = ZImagePipeline.from_pretrained(
                    str(MODEL_PATH),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
                )
                if torch.cuda.is_available():
                    state.pipeline.to("cuda")
                state.pipeline.enable_model_cpu_offload()
            
            # LoRA handling
            if req.lora_path:
                if state.current_lora_path != req.lora_path:
                    if state.current_lora_path:
                        try:
                            state.pipeline.unload_lora_weights()
                        except:
                            pass
                    state.pipeline.load_lora_weights(req.lora_path)
                    state.current_lora_path = req.lora_path
            else:
                if state.current_lora_path:
                    try:
                        state.pipeline.unload_lora_weights()
                    except:
                        pass
                    state.current_lora_path = None
            
            # Seed
            generator = None
            actual_seed = req.seed
            if actual_seed != -1:
                generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
            else:
                generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
                actual_seed = generator.seed()
            
            # Progress callback
            def progress_callback(pipe, step_index, timestep, callback_kwargs):
                step = step_index + 1
                progress_queue.put({"stage": "generating", "step": step, "total": req.steps, "message": f"Step {step}/{req.steps}"})
                return callback_kwargs
            
            progress_queue.put({"stage": "generating", "step": 0, "total": req.steps, "message": "Starting..."})
            
            if req.lora_path:
                state.pipeline.cross_attention_kwargs = {"scale": req.lora_scale}
            
            image = state.pipeline(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=generator,
                callback_on_step_end=progress_callback,
            ).images[0]
            
            if state.pipeline:
                state.pipeline.cross_attention_kwargs = None
            
            progress_queue.put({"stage": "saving", "step": req.steps, "total": req.steps, "message": "Saving..."})
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            generated_dir = OUTPUTS_DIR / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{timestamp}.png"
            image_path = generated_dir / filename
            image.save(image_path)
            
            # Metadata
            metadata = req.dict()
            metadata["timestamp"] = timestamp
            metadata["seed"] = actual_seed
            with open(generated_dir / f"{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            result_holder["result"] = {
                "success": True,
                "image": f"data:image/png;base64,{img_str}",
                "url": f"/outputs/generated/{filename}",
                "seed": actual_seed
            }
            progress_queue.put({"stage": "completed", "step": req.steps, "total": req.steps, "message": "Done!"})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_holder["error"] = str(e)
            progress_queue.put({"stage": "failed", "step": 0, "total": req.steps, "message": str(e)})
    
    async def event_generator():
        # Start generation in thread
        thread = threading.Thread(target=do_generate_with_queue)
        thread.start()
        
        while True:
            try:
                # Non-blocking check
                try:
                    progress = progress_queue.get_nowait()
                    yield f"data: {json.dumps(progress)}\n\n"
                    if progress["stage"] in ("completed", "failed"):
                        # 等待线程完成以确保结果已设置
                        thread.join(timeout=5)
                        # Send final result
                        if result_holder["result"]:
                            yield f"data: {json.dumps(result_holder['result'])}\n\n"
                        elif result_holder["error"]:
                            yield f"data: {json.dumps({'error': result_holder['error']})}\n\n"
                        break
                except queue.Empty:
                    pass
                await asyncio.sleep(0.05)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
        
        thread.join(timeout=1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/generate")
async def generate_image(req: GenerationRequest):
    """Generate image using Z-Image model"""
    import asyncio
    import concurrent.futures
    
    # 立即更新状态（在主线程中），确保轮询能立刻看到
    state.generation_status["running"] = True
    state.generation_status["stage"] = "loading"
    state.generation_status["current_step"] = 0
    state.generation_status["total_steps"] = req.steps
    state.generation_status["progress"] = 0
    state.generation_status["message"] = "Initializing..."
    state.generation_status["error"] = None
    print(f"[Generation] Started - status set to running")
    
    # 在线程池中执行耗时操作
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    def do_generate():
        """在线程中执行的生成函数"""
        from diffusers import ZImagePipeline
        
        # 更新状态：开始加载
        print(f"[Generation] Thread started - loading model")
        sync_broadcast_generation_progress(0, req.steps, "loading", "Loading model...")
        
        # Load model if not loaded
        if state.pipeline is None:
            from core.config import MODEL_PATH
            model_path = MODEL_PATH
            if not model_path.exists():
                raise Exception("Model not found. Please download it first.")
            
            print(f"Loading model from {model_path}...")
            state.pipeline = ZImagePipeline.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
            )
            if torch.cuda.is_available():
                state.pipeline.to("cuda")
            state.pipeline.enable_model_cpu_offload()
            print("Model loaded successfully.")

        # Set seed
        generator = None
        actual_seed = req.seed
        if actual_seed != -1:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
        else:
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
            actual_seed = generator.seed()

        # 进度回调函数
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            """Pipeline 进度回调"""
            total_steps = req.steps
            current_step = step_index + 1
            print(f"[Progress Callback] Step {current_step}/{total_steps}")
            
            # 直接更新状态
            state.generation_status["running"] = True
            state.generation_status["current_step"] = current_step
            state.generation_status["total_steps"] = total_steps
            state.generation_status["progress"] = round(current_step / total_steps * 100, 1)
            state.generation_status["stage"] = "generating"
            state.generation_status["message"] = f"Step {current_step}/{total_steps}"
            
            # 也通过广播发送
            sync_broadcast_generation_progress(
                current_step=current_step,
                total_steps=total_steps,
                stage="generating",
                message=f"Step {current_step}/{total_steps}"
            )
            return callback_kwargs
        
        # Helper for generation
        def run_inference(prompt, seed_generator, lora_scale=0.0):
            # 设置 LoRA 权重 (使用 set_adapters 而不是 cross_attention_kwargs)
            if req.lora_path and state.current_lora_path:
                try:
                    # 获取已加载的 adapter 名称
                    # get_list_adapters() 返回 {'transformer': ['default_0'], ...}
                    # 我们需要提取实际的 adapter 名称列表
                    if hasattr(state.pipeline, 'get_list_adapters'):
                        adapters_dict = state.pipeline.get_list_adapters()
                        # 提取所有 adapter 名称 (通常是 'default_0')
                        adapter_names = []
                        for component_adapters in adapters_dict.values():
                            adapter_names.extend(component_adapters)
                        adapter_names = list(set(adapter_names))  # 去重
                        
                        if adapter_names:
                            # 为每个 adapter 设置相同的权重
                            weights = [lora_scale] * len(adapter_names)
                            state.pipeline.set_adapters(adapter_names, adapter_weights=weights)
                            print(f"[LoRA] Set adapters {adapter_names} scale to {lora_scale}")
                        else:
                            state.pipeline.cross_attention_kwargs = {"scale": lora_scale}
                    else:
                        state.pipeline.cross_attention_kwargs = {"scale": lora_scale}
                except Exception as e:
                    print(f"[LoRA] Failed to set adapter scale: {e}, using cross_attention_kwargs")
                    state.pipeline.cross_attention_kwargs = {"scale": lora_scale}
            
            return state.pipeline(
                prompt=prompt,
                negative_prompt=req.negative_prompt,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                generator=seed_generator,
                callback_on_step_end=progress_callback,
            ).images[0]

        # 智能 LoRA 管理
        if req.lora_path:
            # 需要 LoRA
            if state.current_lora_path != req.lora_path:
                # LoRA 变化了，需要切换
                if state.current_lora_path:
                    # 先卸载旧的 LoRA
                    print(f"Unloading previous LoRA: {state.current_lora_path}")
                    try:
                        state.pipeline.unload_lora_weights()
                    except Exception:
                        pass
                
                # 加载新的 LoRA
                print(f"Loading LoRA: {req.lora_path}")
                try:
                    state.pipeline.load_lora_weights(req.lora_path)
                    state.current_lora_path = req.lora_path
                    # 打印已加载的 adapters
                    if hasattr(state.pipeline, 'get_list_adapters'):
                        adapters = state.pipeline.get_list_adapters()
                        print(f"[LoRA] Loaded adapters: {adapters}")
                except Exception as e:
                    state.current_lora_path = None
                    print(f"Failed to load LoRA: {e}")
                    import traceback
                    traceback.print_exc()
                    raise Exception(f"Failed to load LoRA: {str(e)}")
            else:
                print(f"LoRA already loaded: {req.lora_path}")
        else:
            # 不需要 LoRA
            if state.current_lora_path:
                # 卸载当前 LoRA
                print(f"Unloading LoRA: {state.current_lora_path}")
                try:
                    state.pipeline.unload_lora_weights()
                except Exception:
                    pass
                state.current_lora_path = None

        final_image = None
        sync_broadcast_generation_progress(0, req.steps, "generating", "Starting generation...")
        
        try:
            if req.comparison_mode and req.lora_path:
                print("Generating Comparison: Base vs LoRA")
                sync_broadcast_generation_progress(0, req.steps * 2, "generating", "Generating base image...")
                
                # 1. Generate Base (LoRA scale 0)
                gen_base = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
                base_image = run_inference(req.prompt, gen_base, lora_scale=0.0)
                
                # 2. Generate LoRA (LoRA scale user_defined)
                gen_lora = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(actual_seed)
                lora_image = run_inference(req.prompt, gen_lora, lora_scale=req.lora_scale)
                
                # 3. Stitch
                total_width = base_image.width + lora_image.width
                max_height = max(base_image.height, lora_image.height)
                final_image = Image.new('RGB', (total_width, max_height))
                final_image.paste(base_image, (0, 0))
                final_image.paste(lora_image, (base_image.width, 0))
                
            else:
                # Normal generation
                print(f"Generating image: {req.prompt}")
                scale = req.lora_scale if req.lora_path else 0.0
                final_image = run_inference(req.prompt, generator, lora_scale=scale)

            # Save image and metadata
            sync_broadcast_generation_progress(req.steps, req.steps, "saving", "Saving image...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            generated_dir = OUTPUTS_DIR / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{timestamp}.png"
            image_path = generated_dir / filename
            final_image.save(image_path)
            
            # Save metadata
            metadata = req.dict()
            metadata["timestamp"] = timestamp
            metadata["seed"] = actual_seed
            metadata_path = generated_dir / f"{timestamp}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            image_url = f"/outputs/generated/{filename}"
            
            # Convert to base64
            buffered = io.BytesIO()
            final_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 完成生成
            sync_broadcast_generation_progress(req.steps, req.steps, "completed", "Generation completed!")
            state.generation_status["running"] = False
            state.generation_status["stage"] = "completed"
            
            return {
                "success": True,
                "image": f"data:image/png;base64,{img_str}",
                "url": image_url,
                "seed": actual_seed,
                "message": "Image generated successfully"
            }
            
        finally:
            # 清理 cross_attention_kwargs 和 adapter weights（LoRA 保持加载）
            if state.pipeline:
                state.pipeline.cross_attention_kwargs = None
                # 重置 adapter weights 到默认值 1.0
                if state.current_lora_path and hasattr(state.pipeline, 'get_list_adapters'):
                    try:
                        adapters_dict = state.pipeline.get_list_adapters()
                        adapter_names = []
                        for component_adapters in adapters_dict.values():
                            adapter_names.extend(component_adapters)
                        adapter_names = list(set(adapter_names))
                        if adapter_names:
                            weights = [1.0] * len(adapter_names)
                            state.pipeline.set_adapters(adapter_names, adapter_weights=weights)
                    except Exception:
                        pass
    
    # 在线程池中执行生成
    try:
        result = await loop.run_in_executor(executor, do_generate)
        return result
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        
        # 更新错误状态
        state.generation_status["running"] = False
        state.generation_status["stage"] = "failed"
        state.generation_status["error"] = str(e)
        sync_broadcast_generation_progress(0, 0, "failed", f"Error: {str(e)}")
        
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/generation-status")
async def get_generation_status():
    """获取当前生成进度状态"""
    return state.generation_status.copy()


@router.post("/api/unload-model")
async def unload_model():
    """手动卸载模型释放显存"""
    import gc
    
    if state.pipeline is not None:
        # 卸载 LoRA
        if state.current_lora_path:
            try:
                state.pipeline.unload_lora_weights()
            except Exception:
                pass
            state.current_lora_path = None
        
        # 卸载模型
        del state.pipeline
        state.pipeline = None
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {"success": True, "message": "Model unloaded successfully"}
    
    return {"success": True, "message": "No model loaded"}


@router.get("/api/model-info")
async def get_model_info():
    """获取当前模型加载状态"""
    return {
        "model_loaded": state.pipeline is not None,
        "current_lora": state.current_lora_path
    }


@router.get("/api/history")
async def get_generation_history():
    """Get list of generated images"""
    generated_dir = OUTPUTS_DIR / "generated"
    if not generated_dir.exists():
        return []
        
    history = []
    for metadata_file in sorted(generated_dir.glob("*.json"), reverse=True):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            timestamp = metadata.get("timestamp")
            if not timestamp:
                continue
                
            image_filename = f"{timestamp}.png"
            if (generated_dir / image_filename).exists():
                history.append({
                    "url": f"/outputs/generated/{image_filename}",
                    "thumbnail": f"/outputs/generated/{image_filename}",
                    "metadata": metadata
                })
        except Exception as e:
            print(f"Error reading history file {metadata_file}: {e}")
            continue
            
    return history

@router.post("/api/history/delete")
async def delete_history(req: DeleteHistoryRequest):
    """Delete generated images and metadata"""
    generated_dir = OUTPUTS_DIR / "generated"
    deleted_count = 0
    errors = []
    
    for timestamp in req.timestamps:
        try:
            image_path = generated_dir / f"{timestamp}.png"
            if image_path.exists():
                image_path.unlink()
                
            metadata_path = generated_dir / f"{timestamp}.json"
            if metadata_path.exists():
                metadata_path.unlink()
                
            deleted_count += 1
        except Exception as e:
            errors.append(f"Failed to delete {timestamp}: {str(e)}")
            
    return {
        "success": True,
        "deleted": deleted_count,
        "errors": errors
    }
