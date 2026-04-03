"""
RunPod Serverless Handler for LTX-Video (Image-to-Video)
"""

import os
import sys
import time
import base64
import tempfile
import traceback
from io import BytesIO

import runpod
import torch
from PIL import Image

PIPE = None
MODEL_ID = "Lightricks/LTX-Video-0.9.7"
VOLUME_PATH = "/runpod-volume"
MODEL_CACHE = os.path.join(VOLUME_PATH, "huggingface_cache")


def load_model():
    global PIPE
    if PIPE is not None:
        return PIPE

    print(f"[LTX Worker] Loading model... cache: {MODEL_CACHE}")
    os.environ["HF_HOME"] = MODEL_CACHE
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE

    from diffusers import LTXImageToVideoPipeline

    PIPE = LTXImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
    )
    PIPE.to("cuda")

    try:
        PIPE.enable_model_cpu_offload()
    except Exception:
        pass

    print("[LTX Worker] Model loaded!")
    return PIPE


def decode_image(image_input):
    if image_input.startswith(("http://", "https://")):
        import requests
        resp = requests.get(image_input, timeout=60)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        data = base64.b64decode(image_input)
        return Image.open(BytesIO(data)).convert("RGB")


def encode_video_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event):
    try:
        inp = event.get("input", {})

        image_input = inp.get("image")
        if not image_input:
            return {"error": "Missing 'image' (base64 or URL)"}

        prompt = inp.get("prompt", "Subtle natural motion, anime style, cinematic lighting")
        negative_prompt = inp.get("negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text")
        num_frames = inp.get("num_frames", 121)
        height = inp.get("height", 512)
        width = inp.get("width", 768)
        steps = inp.get("num_inference_steps", 30)
        guidance = inp.get("guidance_scale", 7.5)
        seed = inp.get("seed", 42)
        fps = inp.get("fps", 24)

        pipe = load_model()

        print(f"[LTX] Generating: {num_frames}f @ {width}x{height}")
        image = decode_image(image_input).resize((width, height), Image.LANCZOS)

        start = time.time()
        gen = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None

        from diffusers.utils import export_to_video
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )

        frames = output.frames[0]
        elapsed = time.time() - start
        print(f"[LTX] Done in {elapsed:.1f}s")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=fps)
        video_b64 = encode_video_to_base64(tmp_path)
        os.unlink(tmp_path)

        return {
            "video_base64": video_b64,
            "duration_seconds": round(len(frames) / fps, 2),
            "num_frames": len(frames),
            "generation_time_seconds": round(elapsed, 1),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print("[LTX Worker] Starting...")
    load_model()
    runpod.serverless.start({"handler": handler})
