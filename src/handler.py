"""
RunPod Serverless Handler for LTX-2.3 Pro (Image-to-Video)
Converts images to short animated videos using LTX-2.3 two-stage pipeline.
"""

import os
import sys
import time
import base64
import tempfile
import traceback
from io import BytesIO
from pathlib import Path

import runpod
import torch
from PIL import Image

# Global: load model once, reuse across requests
PIPELINE = None
VOLUME_PATH = "/runpod-volume"
MODEL_CACHE = os.path.join(VOLUME_PATH, "huggingface_cache")
MODELS_DIR = os.path.join(VOLUME_PATH, "ltx_models")


def download_models():
    """Download LTX-2.3 model files from HuggingFace if not cached."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_CACHE, exist_ok=True)

    from huggingface_hub import hf_hub_download

    model_files = {
        "checkpoint": ("Lightricks/LTX-2.3", "ltx-2.3-22b-dev-fp8.safetensors"),
        "distilled_lora": ("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors"),
        "spatial_upsampler": ("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    }

    paths = {}
    for key, (repo_id, filename) in model_files.items():
        local_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(local_path):
            print(f"[LTX] {filename} -- already cached")
            paths[key] = local_path
        else:
            print(f"[LTX] Downloading {filename}...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=MODELS_DIR,
                cache_dir=MODEL_CACHE,
            )
            paths[key] = downloaded
            print(f"[LTX] {filename} -- downloaded!")

    # Download Gemma 3 text encoder
    gemma_dir = os.path.join(MODELS_DIR, "gemma-3-4b-it")
    if os.path.exists(gemma_dir) and any(f.endswith(".safetensors") for f in os.listdir(gemma_dir)):
        print("[LTX] Gemma 3 -- already cached")
        paths["gemma_root"] = gemma_dir
    else:
        print("[LTX] Downloading Gemma 3 4B IT...")
        from huggingface_hub import snapshot_download
        gemma_dir = snapshot_download(
            "google/gemma-3-4b-it",
            local_dir=gemma_dir,
            cache_dir=MODEL_CACHE,
            ignore_patterns=["*.gguf", "*.bin"],
        )
        paths["gemma_root"] = gemma_dir
        print("[LTX] Gemma 3 -- downloaded!")

    return paths


def load_model():
    """Load the LTX-2.3 two-stage pipeline (called once at worker startup)."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("[LTX Worker] Loading LTX-2.3 Pro model...")
    os.environ["HF_HOME"] = MODEL_CACHE
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

    paths = download_models()

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            paths["distilled_lora"],
            0.6,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ]

    PIPELINE = TI2VidTwoStagesPipeline(
        checkpoint_path=paths["checkpoint"],
        distilled_lora=distilled_lora,
        spatial_upsampler_path=paths["spatial_upsampler"],
        gemma_root=paths["gemma_root"],
        loras=[],
        fp8transformer=True,
        offload_transformer=True,
    )

    print("[LTX Worker] LTX-2.3 Pro model loaded!")
    return PIPELINE


def decode_image(image_input):
    """Decode image from base64 or URL and save to temp file."""
    tmp_path = tempfile.mktemp(suffix=".jpg")

    if image_input.startswith(("http://", "https://")):
        import requests
        resp = requests.get(image_input, timeout=60)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        data = base64.b64decode(image_input)
        img = Image.open(BytesIO(data)).convert("RGB")

    img.save(tmp_path, "JPEG", quality=90)
    return tmp_path


def handler(event):
    """
    Input:  { "input": { "image": "<base64_or_url>", "prompt": "...", ... } }
    Output: { "video_base64": "...", "duration_seconds": ..., ... }
    """
    try:
        inp = event.get("input", {})

        image_input = inp.get("image")
        if not image_input:
            return {"error": "Missing 'image' (base64 or URL)"}

        prompt = inp.get("prompt",
            "Subtle natural motion, anime style, cinematic lighting, gentle wind, soft camera movement")
        negative_prompt = inp.get("negative_prompt",
            "static image, tilted camera, unnatural transitions, jittery movement")
        num_frames = inp.get("num_frames", 97)
        height = inp.get("height", 512)
        width = inp.get("width", 768)
        steps = inp.get("num_inference_steps", 40)
        cfg_scale = inp.get("cfg_guidance_scale", 4.0)
        seed = inp.get("seed", 42)
        fps = inp.get("frame_rate", 24)
        image_strength = inp.get("image_strength", 0.8)

        pipeline = load_model()
        image_path = decode_image(image_input)

        print(f"[LTX] Generating: {num_frames}f @ {width}x{height}, steps={steps}")
        start = time.time()

        from ltx_core.components.guiders import MultiModalGuiderParams

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )

        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[29],
        )

        output_path = tempfile.mktemp(suffix=".mp4")

        pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=output_path,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            num_inference_steps=steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[(image_path, 0, image_strength)],
        )

        elapsed = time.time() - start
        print(f"[LTX] Done in {elapsed:.1f}s")

        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        os.unlink(image_path)
        os.unlink(output_path)

        return {
            "video_base64": video_b64,
            "duration_seconds": round(num_frames / fps, 2),
            "num_frames": num_frames,
            "generation_time_seconds": round(elapsed, 1),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print("[LTX Worker] Starting LTX-2.3 Pro worker...")
    load_model()
    runpod.serverless.start({"handler": handler})
