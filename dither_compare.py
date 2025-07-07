from PIL import Image
import os

def load_image(filename):
    return Image.open(filename).convert("RGB")

def reduce_colors(img, num_colors=50, use_dither=True):
    """
    Reduce image to a specific number of colors with optional dithering.
    """
    dither_mode = Image.Dither.FLOYDSTEINBERG if use_dither else Image.Dither.NONE
    return img.quantize(colors=num_colors, method=Image.MEDIANCUT, dither=dither_mode)

def apply_variants(img, output_dir="compared_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    variants = [
        {"colors": 50, "dither": False, "name": "50_colors_no_dither"},
        {"colors": 50, "dither": True,  "name": "50_colors_dither"},
        {"colors": 20, "dither": True,  "name": "20_colors_dither"},
    ]

    for variant in variants:
        result = reduce_colors(img, variant["colors"], use_dither=variant["dither"])
        result = result.convert("RGB")  # convert back for display/consistency
        result.save(os.path.join(output_dir, f"{variant['name']}.png"))
        print(f"✅ Saved: {variant['name']}.png")

if __name__ == "__main__":
    try:
        img = load_image("twotwo.png")  # or whatever your actual image filename is
        apply_variants(img)
        print("\n🎉 All variants saved in 'compared_outputs/'")
    except Exception as e:
        print(f"❌ Error: {e}")
