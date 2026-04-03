"""
Batch Image-to-Video Generator using RunPod Serverless API
Converts multiple anime images to short videos using LTX-Video.

Usage:
    python batch_generate.py --input_dir ./images --output_dir ./videos
    python batch_generate.py --input_dir ./images --output_dir ./videos --concurrent 3
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "h883b7g7a42yoz")
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

# Default generation settings
DEFAULT_PROMPT = "Subtle natural motion, anime style, cinematic lighting, gentle wind blowing hair, soft camera movement"
DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, watermark, text, static, no motion"
DEFAULT_FRAMES = 121  # ~5 seconds at 24fps
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 768
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
DEFAULT_FPS = 24

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}


def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def submit_job(image_path: str, prompt: str = DEFAULT_PROMPT, seed: int = -1) -> dict:
    """Submit an async job to RunPod endpoint."""
    image_b64 = image_to_base64(image_path)

    payload = {
        "input": {
            "image": image_b64,
            "prompt": prompt,
            "negative_prompt": DEFAULT_NEGATIVE,
            "num_frames": DEFAULT_FRAMES,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
            "num_inference_steps": DEFAULT_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE,
            "fps": DEFAULT_FPS,
            "seed": seed,
        }
    }

    resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    return result


def check_status(job_id: str) -> dict:
    """Check the status of a submitted job."""
    resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_for_job(job_id: str, image_name: str, poll_interval: int = 5, timeout: int = 600) -> dict:
    """Poll until job completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        result = check_status(job_id)
        status = result.get("status")

        if status == "COMPLETED":
            print(f"  ✅ {image_name} — completed!")
            return result
        elif status == "FAILED":
            error = result.get("error", "Unknown error")
            print(f"  ❌ {image_name} — failed: {error}")
            return result
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            elapsed = int(time.time() - start)
            print(f"  ⏳ {image_name} — {status} ({elapsed}s)", end="\r")

        time.sleep(poll_interval)

    print(f"  ⏰ {image_name} — timed out after {timeout}s")
    return {"status": "TIMEOUT", "error": "Job timed out"}


def save_video(video_base64: str, output_path: str):
    """Decode base64 video and save to file."""
    video_data = base64.b64decode(video_base64)
    with open(output_path, "wb") as f:
        f.write(video_data)


def process_single_image(image_path: str, output_dir: str, prompt: str, seed: int) -> dict:
    """Process a single image: submit, wait, save."""
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}.mp4")

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"  ⏭️  {image_name} — already exists, skipping")
        return {"status": "SKIPPED", "image": image_name}

    try:
        # Submit job
        result = submit_job(image_path, prompt=prompt, seed=seed)
        job_id = result.get("id")

        if not job_id:
            print(f"  ❌ {image_name} — no job ID returned: {result}")
            return {"status": "FAILED", "image": image_name, "error": "No job ID"}

        print(f"  🚀 {image_name} — submitted (job: {job_id})")

        # Wait for completion
        final = wait_for_job(job_id, image_name)

        if final.get("status") == "COMPLETED":
            output_data = final.get("output", {})
            video_b64 = output_data.get("video_base64")

            if video_b64:
                save_video(video_b64, output_path)
                duration = output_data.get("duration_seconds", "?")
                gen_time = output_data.get("generation_time_seconds", "?")
                print(f"  💾 {image_name} — saved ({duration}s video, generated in {gen_time}s)")
                return {
                    "status": "SUCCESS",
                    "image": image_name,
                    "output": output_path,
                    "duration": duration,
                    "gen_time": gen_time,
                }
            else:
                print(f"  ❌ {image_name} — no video in response")
                return {"status": "FAILED", "image": image_name, "error": "No video data"}
        else:
            return {"status": "FAILED", "image": image_name, "error": final.get("error", "Unknown")}

    except Exception as e:
        print(f"  ❌ {image_name} — error: {e}")
        return {"status": "FAILED", "image": image_name, "error": str(e)}


def batch_process(input_dir: str, output_dir: str, prompt: str, concurrent: int, seed: int):
    """Process all images in directory with concurrent workers."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if Path(f).suffix.lower() in image_extensions
    ])

    if not images:
        print(f"❌ No images found in {input_dir}")
        return

    total = len(images)
    print(f"\n{'='*60}")
    print(f"🎬 LTX-Video Batch Generator")
    print(f"{'='*60}")
    print(f"📁 Input:  {input_dir} ({total} images)")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Workers: {concurrent} concurrent")
    print(f"🎯 Prompt: {prompt[:80]}...")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(process_single_image, img, output_dir, prompt, seed): img
            for img in images
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            # Progress
            success = sum(1 for r in results if r["status"] == "SUCCESS")
            failed = sum(1 for r in results if r["status"] == "FAILED")
            skipped = sum(1 for r in results if r["status"] == "SKIPPED")
            print(f"\n  📊 Progress: {i}/{total} | ✅ {success} | ❌ {failed} | ⏭️ {skipped}")

    # Summary
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")

    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Success: {success_count}/{total}")
    print(f"❌ Failed:  {failed_count}/{total}")
    print(f"⏭️  Skipped: {skipped_count}/{total}")
    print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if success_count > 0:
        avg_time = elapsed / success_count
        print(f"⚡ Avg per video: {avg_time:.1f}s")

    # Save results log
    log_path = os.path.join(output_dir, "batch_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": total,
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_time_seconds": round(elapsed, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"📝 Log saved: {log_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch Image-to-Video using RunPod LTX-Video")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output videos")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Motion prompt for all videos")
    parser.add_argument("--concurrent", type=int, default=3, help="Number of concurrent workers (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")

    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("❌ RUNPOD_API_KEY environment variable is not set")
        print("   export RUNPOD_API_KEY='your_api_key_here'")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        concurrent=args.concurrent,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
