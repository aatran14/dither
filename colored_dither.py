from PIL import Image
import numpy as np
import os

def load_image(filename):
    return Image.open(os.path.join(os.getcwd(), filename)).convert("RGB")

def stippled_color_dither(img, block_size=2, palette_size=16, noise_strength=30):
    """
    Applies particle-like stippled color dithering using block averaging and noise perturbation.
    :param img: RGB image
    :param block_size: Block size for coarse averaging
    :param palette_size: Number of output colors
    :param noise_strength: Random offset strength added to color values
    :return: Dithered image (mode RGB)
    """
    arr = np.array(img)
    h, w, _ = arr.shape
    h = h - (h % block_size)
    w = w - (w % block_size)
    arr = arr[:h, :w]

    # Add random noise (simulate particles)
    noisy_arr = arr + np.random.randint(-noise_strength, noise_strength + 1, arr.shape)
    noisy_arr = np.clip(noisy_arr, 0, 255)

    # Downsample by block averaging
    small = noisy_arr.reshape(
        h // block_size, block_size,
        w // block_size, block_size, 3
    ).mean(axis=(1,3)).astype(np.uint8)

    # Quantize small image to limited palette
    small_img = Image.fromarray(small, mode="RGB")
    small_quant = small_img.quantize(colors=palette_size, method=Image.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)

    # Upsample to original size using nearest (preserve blocks)
    result = small_quant.convert("RGB").resize((w, h), resample=Image.Resampling.NEAREST)

    return result

def apply_colored_stipple(img, output_dir="color_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {"block_size": 2, "palette_size": 16, "noise": 20},
        {"block_size": 3, "palette_size": 8,  "noise": 40},
        {"block_size": 4, "palette_size": 6,  "noise": 50},
    ]

    for i, cfg in enumerate(configs, 1):
        print(f"🟡 Particle Dither {i}: blocks={cfg['block_size']} colors={cfg['palette_size']} noise={cfg['noise']}")
        result = stippled_color_dither(img, cfg["block_size"], cfg["palette_size"], cfg["noise"])
        out_path = os.path.join(output_dir, f"stippled_{i}.png")
        result.save(out_path)

    print(f"\n Particle-styled color dithers saved in '{output_dir}'.")

if __name__ == "__main__":
    filename = input("Enter image filename (in same folder): ").strip()
    try:
        img = load_image(filename)
        apply_colored_stipple(img)
    except Exception as e:
        print(f"Error: {e}")
