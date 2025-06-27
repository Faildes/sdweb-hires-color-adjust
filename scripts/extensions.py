import numpy as np
from PIL import Image
from modules import scripts, images
import gradio as gr

# Color balance function as provided
def color_balance(img: Image.Image, adj: dict) -> Image.Image:
    for k in ('shadow', 'middle', 'highlight'):
        v = adj.get(k)
        if not isinstance(v, (list, tuple)) or len(v) != 5:
            raise ValueError(f"adjustments['{k}'] must be a length-5 list or tuple")
    mode = img.mode
    if mode == 'RGBA':
        rgb, alpha = img.convert('RGB'), img.split()[-1]
    else:
        alpha = None
        rgb = img.convert('RGB') if mode != 'RGB' else img
    orig = np.array(rgb, dtype=np.float32)
    lum = orig.mean(axis=2)
    ws = np.clip((128.0 - lum) / 128.0, 0.0, 1.0)
    wh = np.clip((lum - 128.0) / 128.0, 0.0, 1.0)
    wm = 1.0 - ws - wh
    res = np.zeros_like(orig)
    for w, region in zip((ws, wm, wh), ('shadow', 'middle', 'highlight')):
        r, g, b, bright, contrast = adj[region]
        af = np.array([r, g, b], dtype=np.float32) / 100.0
        delta = (255.0 - orig) * np.maximum(af, 0.0) + orig * np.minimum(af, 0.0)
        rv = (orig + delta) * bright
        mean = rv.mean(axis=(0,1), keepdims=True)
        rv = (rv - mean) * contrast + mean
        rv = np.clip(rv, 0, 255)
        res += rv * w[..., None]
    out = Image.fromarray(np.clip(res, 0.0, 255.0).astype(np.uint8), 'RGB')
    if alpha is not None:
        out = Image.merge('RGBA', (*out.split(), alpha))
    del orig, res, rv, delta, lum, ws, wh, wm
    return out

class Script(scripts.Script):
    def title(self):
        return "Hires Color Adjustment"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            gr.Markdown("### Hires Color Adjustment")
            sliders = {}
            for region in ('shadow', 'middle', 'highlight'):
                with gr.Row(equal_height=True):
                    r = gr.Slider(-100, 100, step=1, value=0, label=f"{region.capitalize()} R")
                    g = gr.Slider(-100, 100, step=1, value=0, label=f"{region.capitalize()} G")
                    b = gr.Slider(-100, 100, step=1, value=0, label=f"{region.capitalize()} B")
                    bright = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label=f"{region.capitalize()} Brightness")
                    contrast = gr.Slider(0.0, 10.0, step=0.1, value=1.0, label=f"{region.capitalize()} Contrast")
                sliders[region] = (r, g, b, bright, contrast)
            # Flatten slider values in order
        return tuple([enabled] + [s for region in sliders.values() for s in region])

    def run(self, p, *args):
        # Build adjustment dict from slider args
        if not args.pop(0): return p
        adjust = {
            'shadow': list(args[0:5]),
            'middle': list(args[5:10]),
            'highlight': list(args[10:15])
        }
        # Apply to initial images before hi-res fix
        new_images = []
        for img in p.init_images:
            pil = images.img_to_pil(img)
            out = color_balance(pil, adjust)
            new_images.append(images.pil_to_image(out))
        p.init_images = new_images
        return p
