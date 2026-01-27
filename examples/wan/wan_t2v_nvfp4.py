"""
Wan2.1 text-to-video generation example (NVFP4).
This example demonstrates how to use LightX2V with Wan2.1 model for T2V generation.
"""

import asyncio
import os
import threading

from lightx2v import LightX2VPipeline
from lightx2v.shared_queue import GENERATION_LOCK, generation_slot

lightx2v_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = os.path.join(lightx2v_root, "models", "Wan2.1-T2V-1.3B")
quantized_ckpt = os.path.join(
    model_path, "wan2.1_t2v_1_3b_nvfp4_lightx2v_4step.safetensors"
)

DEFAULT_SEED = 42
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely "
    "on a spotlighted stage."
)
DEFAULT_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
DEFAULT_SAVE_NAME = "output.mp4"

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def _create_pipeline() -> LightX2VPipeline:
    pipe = LightX2VPipeline(
        model_path=model_path,
        model_cls="wan2.1_distill",
        task="t2v",
    )

    # Alternative: create generator from config JSON file
    # pipe.create_generator(config_json="../configs/wan/wan_t2v.json")

    pipe.enable_quantize(
        dit_quantized=True,
        dit_quantized_ckpt=quantized_ckpt,
        quant_scheme="nvfp4",
    )

    # Create generator with specified parameters
    pipe.create_generator(
        attn_mode="sage_attn2",
        infer_steps=4,
        height=480,  # Can be set to 720 for higher resolution
        width=832,  # Can be set to 1280 for higher resolution
        num_frames=81,
        guidance_scale=1.0,
        sample_shift=5.0,
    )
    return pipe


def get_pipeline() -> LightX2VPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        with _PIPELINE_LOCK:
            if _PIPELINE is None:
                _PIPELINE = _create_pipeline()
    return _PIPELINE


def generate_video(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = DEFAULT_SEED,
    save_result_path: str | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt is required")

    if save_result_path is None:
        save_dir = os.path.join(lightx2v_root, "save_results")
        os.makedirs(save_dir, exist_ok=True)
        save_result_path = os.path.join(save_dir, DEFAULT_SAVE_NAME)

    pipe = get_pipeline()
    with GENERATION_LOCK:
        pipe.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=save_result_path,
        )
    return save_result_path


async def generate_video_async(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = DEFAULT_SEED,
    save_result_path: str | None = None,
    request_id: str | None = None,
) -> str:
    async with generation_slot(request_id=request_id):
        return await asyncio.to_thread(
            generate_video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            save_result_path=save_result_path,
        )


def main() -> None:
    save_result_path = os.path.join(lightx2v_root, "save_results", DEFAULT_SAVE_NAME)
    generate_video(
        seed=DEFAULT_SEED,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        save_result_path=save_result_path,
    )


if __name__ == "__main__":
    main()
