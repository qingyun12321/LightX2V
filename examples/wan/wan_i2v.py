"""
Wan2.1 image-to-video generation example.
This example demonstrates how to use LightX2V with Wan2.1 model for I2V generation.
"""

import asyncio
import os
import threading

from lightx2v import LightX2VPipeline
from lightx2v.shared_queue import GENERATION_LOCK, generation_slot

model_path = os.path.join("models", "Wan2.1-I2V-14B-480P")

DEFAULT_SEED = 42
DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds. The cat assumes a naturally "
    "relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot "
    "highlights the feline's intricate details and the refreshing atmosphere of the seaside."
)
DEFAULT_NEGATIVE_PROMPT = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
DEFAULT_IMAGE_PATH = os.path.join("assets", "inputs", "imgs", "img_0.jpg")
DEFAULT_SAVE_NAME = "output_i2v.mp4"

_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()


def _create_pipeline() -> LightX2VPipeline:
    pipe = LightX2VPipeline(
        model_path=model_path,
        model_cls="wan2.1",
        task="i2v",
        # Use the full-precision shard directory (index.json) to avoid mixing quantized weights.
        dit_original_ckpt=model_path,
    )

    # Alternative: create generator from config JSON file
    # pipe.create_generator(config_json="../configs/wan/wan_i2v.json")

    # Enable offloading to significantly reduce VRAM usage with minimal speed impact
    # Suitable for RTX 30/40/50 consumer GPUs
    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",  # For Wan models, supports both "block" and "phase"
        text_encoder_offload=True,
        image_encoder_offload=False,
        vae_offload=False,
    )

    # Create generator manually with specified parameters
    pipe.create_generator(
        attn_mode="sage_attn2",
        infer_steps=40,
        height=480,  # Can be set to 720 for higher resolution
        width=832,  # Can be set to 1280 for higher resolution
        num_frames=81,
        guidance_scale=5.0,
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
    image_path: str,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = DEFAULT_SEED,
    save_result_path: str | None = None,
) -> str:
    if not image_path:
        raise ValueError("image_path is required")
    if not prompt:
        raise ValueError("prompt is required")

    if save_result_path is None:
        save_dir = os.path.join("save_results")
        os.makedirs(save_dir, exist_ok=True)
        save_result_path = os.path.join(save_dir, DEFAULT_SAVE_NAME)

    pipe = get_pipeline()
    with GENERATION_LOCK:
        pipe.generate(
            seed=seed,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=save_result_path,
        )
    return save_result_path


async def generate_video_async(
    image_path: str,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = DEFAULT_SEED,
    save_result_path: str | None = None,
    request_id: str | None = None,
) -> str:
    async with generation_slot(request_id=request_id):
        return await asyncio.to_thread(
            generate_video,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            save_result_path=save_result_path,
        )


def main() -> None:
    save_result_path = os.path.join("save_results", DEFAULT_SAVE_NAME)
    generate_video(
        seed=DEFAULT_SEED,
        image_path=DEFAULT_IMAGE_PATH,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        save_result_path=save_result_path,
    )


if __name__ == "__main__":
    main()
