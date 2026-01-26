import argparse
import importlib.util
import os
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from lightx2v.shared_queue import get_queue_status

WAN_T2V_PATH = os.path.join(PROJECT_ROOT, "examples", "wan", "wan_t2v.py")
WAN_I2V_PATH = os.path.join(PROJECT_ROOT, "examples", "wan", "wan_i2v.py")
SAVE_DIR = os.path.join(PROJECT_ROOT, "save_results")
UPLOAD_DIR = os.path.join(SAVE_DIR, "uploads")


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_name} module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan_t2v = _load_module("wan_t2v", WAN_T2V_PATH)
wan_i2v = _load_module("wan_i2v", WAN_I2V_PATH)

app = FastAPI(title="LightX2V X2V API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/queue_status")
def queue_status(request_id: str | None = None):
    return get_queue_status(request_id=request_id)


@app.post("/generate")
async def generate_video(
    prompt: str = Form(default=""),
    negative_prompt: str = Form(default=""),
    seed: int | None = Form(default=None),
    request_id: str = Form(default=""),
    image: UploadFile | None = File(default=None),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    request_id = request_id.strip() or None

    os.makedirs(SAVE_DIR, exist_ok=True)
    if image is None:
        safe_negative = (
            negative_prompt.strip()
            if negative_prompt.strip()
            else wan_t2v.DEFAULT_NEGATIVE_PROMPT
        )
        safe_seed = seed if seed is not None else wan_t2v.DEFAULT_SEED
        filename = f"t2v_{uuid.uuid4().hex}.mp4"
        save_path = os.path.join(SAVE_DIR, filename)
        try:
            output_path = await wan_t2v.generate_video_async(
                prompt=prompt,
                negative_prompt=safe_negative,
                seed=safe_seed,
                save_result_path=save_path,
                request_id=request_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
    else:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        ext = os.path.splitext(image.filename or "")[1] or ".png"
        upload_name = f"upload_{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, upload_name)
        try:
            with open(upload_path, "wb") as f:
                while True:
                    chunk = await image.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        finally:
            await image.close()

        safe_negative = (
            negative_prompt.strip()
            if negative_prompt.strip()
            else wan_i2v.DEFAULT_NEGATIVE_PROMPT
        )
        safe_seed = seed if seed is not None else wan_i2v.DEFAULT_SEED
        filename = f"i2v_{uuid.uuid4().hex}.mp4"
        save_path = os.path.join(SAVE_DIR, filename)
        try:
            output_path = await wan_i2v.generate_video_async(
                image_path=upload_path,
                prompt=prompt,
                negative_prompt=safe_negative,
                seed=safe_seed,
                save_result_path=save_path,
                request_id=request_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")
        finally:
            if os.path.exists(upload_path):
                os.remove(upload_path)

    return FileResponse(output_path, media_type="video/mp4", filename=filename)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightX2V X2V API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=10085, help="Bind port")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
