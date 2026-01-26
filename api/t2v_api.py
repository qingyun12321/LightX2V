import argparse
import importlib.util
import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

SCRIPT_DIR = os.path.dirname(__file__)
os.chdir(SCRIPT_DIR)
sys.path.insert(0, os.path.join(".."))

WAN_T2V_PATH = os.path.join("..", "examples", "wan", "wan_t2v.py")
SAVE_DIR = os.path.join("..", "save_results")


def _load_wan_t2v():
    spec = importlib.util.spec_from_file_location("wan_t2v", WAN_T2V_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load wan_t2v module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wan_t2v = _load_wan_t2v()

app = FastAPI(title="LightX2V T2V API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    seed: int | None = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate")
def generate_video(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    negative_prompt = (
        req.negative_prompt.strip()
        if req.negative_prompt and req.negative_prompt.strip()
        else wan_t2v.DEFAULT_NEGATIVE_PROMPT
    )
    seed = req.seed if req.seed is not None else wan_t2v.DEFAULT_SEED

    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"t2v_{uuid.uuid4().hex}.mp4"
    save_path = os.path.join(SAVE_DIR, filename)

    try:
        output_path = wan_t2v.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            save_result_path=save_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=filename,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LightX2V T2V API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=10085, help="Bind port")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
