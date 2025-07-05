import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import os

# ── optional: keep cache off C: drive when you test locally ───────────
# os.environ["HF_HOME"] = "E:/huggingface_cache"

# ── choose device ─────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype   = torch.float16 if device == "cuda" else torch.float32

# ── load Stable Diffusion v1-5 (photorealistic) ───────────────────────
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)
pipe = pipe.to(device)

# ── generation function ───────────────────────────────────────────────
def generate(prompt: str):
    # quick safety-guard for empty prompts
    if not prompt or prompt.isspace():
        return None
    # run inference
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    return image

# ── Gradio UI ─────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Enter your prompt",
                      placeholder="e.g. a photorealistic lion in the jungle at sunset"),
    outputs=gr.Image(type="pil"),
    title="Stable Bud – Text → Photorealistic Image",
    description=("Powered by Stable Diffusion v1.5 – switch your Space to a GPU (T4) "
                 "for images in ~10 s.  CPU fallback works but is slow.")
)

# ── launch ────────────────────────────────────────────────────────────
demo.launch()