from PIL import Image, ImageOps
import numpy as np
import os
import random

def load_image(filename):
    return Image.open(os.path.join(os.getcwd(), filename)).convert("RGB")

def to_grayscale(img):
    return img.convert("L")

# --- Dithering Methods ---

def no_dither(image):
    return image.convert("1", dither=Image.Dither.NONE)

def floyd_steinberg(image):
    return image.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

def posterize_dither(image):
    return ImageOps.posterize(image.convert("L"), 2)

def bayer_dither(image):
    bayer_matrix = np.array([
    [0, 2],
    [3, 1]
]) / 4.0

    grayscale = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = grayscale.shape
    threshold_map = np.tile(bayer_matrix, (h // 4 + 1, w // 4 + 1))[:h, :w]
    dithered = (grayscale > threshold_map) * 255
    return Image.fromarray(dithered.astype(np.uint8), mode="L")

def random_dither(image):
    grayscale = np.array(image.convert("L"), dtype=np.uint8)
    noise = np.random.randint(0, 256, grayscale.shape)
    dithered = (grayscale > noise) * 255
    return Image.fromarray(dithered.astype(np.uint8), mode="L")

def atkinson_dither(image):
    img = np.array(image.convert("L"), dtype=np.uint8)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 0 if old < 128 else 255
            img[y, x] = new
            error = old - new
            for dx, dy in [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] = np.clip(img[ny, nx] + error * 1/8, 0, 255)
    return Image.fromarray(img.astype(np.uint8), mode="L")

# --- Apply All ---

def apply_all_dithers(img, output_dir="dithered_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "no_dither.png": no_dither,
        "floyd_steinberg.png": floyd_steinberg,
        "posterize.png": posterize_dither,
        "bayer.png": bayer_dither,
        "random.png": random_dither,
        "atkinson.png": atkinson_dither
    }

    for filename, method in methods.items():
        print(f"Processing: {filename}")
        result = method(img)
        result.save(os.path.join(output_dir, filename))

    print(f"\n✅ Dithered images saved in '{output_dir}'.")

# --- Main ---

if __name__ == "__main__":
    filename = input("Enter the image filename (in same folder): ").strip()
    try:
        image = load_image(filename)
        apply_all_dithers(image)
    except Exception as e:
        print(f"❌ Error: {e}")
