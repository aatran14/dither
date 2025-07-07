from PIL import Image, ImageOps, ImageDraw
import numpy as np
import os

def load_image(filename):
    return Image.open(os.path.join(os.getcwd(), filename)).convert("RGB")

def to_grayscale(img):
    return img.convert("L")

def no_dither(img):
    return img.convert("1", dither=Image.Dither.NONE)

def floyd_steinberg(img):
    return img.convert("1", dither=Image.Dither.FLOYDSTEINBERG)

def posterize_1bit(img):
    return ImageOps.posterize(to_grayscale(img), 1)

def posterize_2bit(img):
    return ImageOps.posterize(to_grayscale(img), 2)

# def bayer_dither(img, matrix):
#     grayscale = np.array(to_grayscale(img), dtype=np.float32) / 255.0
#     h, w = grayscale.shape
#     thres_map = np.tile(matrix, (h // matrix.shape[0] + 1, w // matrix.shape[1] + 1))[:h, :w]
#     dithered = (grayscale > thres_map) * 255
#     return Image.fromarray(dithered.astype(np.uint8), mode="L")

def bayer_dither(img, matrix, bias=0.1):
    grayscale = np.array(to_grayscale(img), dtype=np.float32) / 255.0
    h, w = grayscale.shape
    threshold_map = np.tile(matrix, (h // matrix.shape[0] + 1, w // matrix.shape[1] + 1))[:h, :w]
    
    # Apply bias to darken
    dithered = ((grayscale - bias) > threshold_map) * 255
    return Image.fromarray(dithered.astype(np.uint8), mode="L")


def random_dither_pixel(img):
    grayscale = np.array(to_grayscale(img), dtype=np.uint8)
    noise = np.random.randint(0, 256, grayscale.shape)
    dithered = (grayscale > noise) * 255
    return Image.fromarray(dithered.astype(np.uint8), mode="L")

def random_dither_blocky(img, block_size=5):
    grayscale = np.array(to_grayscale(img), dtype=np.uint8)
    h, w = grayscale.shape
    small_noise = np.random.randint(0, 256, (h // block_size + 1, w // block_size + 1))
    noise = np.kron(small_noise, np.ones((block_size, block_size)))[:h, :w]
    dithered = (grayscale > noise) * 255
    return Image.fromarray(dithered.astype(np.uint8), mode="L")

def atkinson(img, reduced=False):
    data = np.array(to_grayscale(img), dtype=np.int16)

    # data = np.array(to_grayscale(img), dtype=np.uint8)
    h, w = data.shape
    spread = [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)] if not reduced else [(1, 0), (0, 1)]
    for y in range(h):
        for x in range(w):
            old = data[y, x]
            new = 255 if old > 128 else 0
            err = old - new
            data[y, x] = new
            for dx, dy in spread:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    data[ny, nx] = np.clip(data[ny, nx] + err * 1/8, 0, 255)
    # return Image.fromarray(data.astype(np.uint8), mode="L")
    return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), mode="L")


def halftone_dither(img, block_size=8):
    """
    Applies a halftone-style circular dot dithering effect.
    Returns a grayscale image with dot intensity based on brightness.
    """
    grayscale = to_grayscale(img)
    w, h = grayscale.size
    output = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(output)
    pixels = np.array(grayscale)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            region = pixels[y:y+block_size, x:x+block_size]
            avg = np.mean(region) if region.size else 255
            radius = (1 - avg / 255.0) * (block_size / 2)
            cx = x + block_size // 2
            cy = y + block_size // 2
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=0
            )

    return output


def custom_3x3_dither(img):
    matrix = np.array([
        [6, 8, 4],
        [1, 0, 3],
        [5, 2, 7]
    ]) / 9.0
    return bayer_dither(img, matrix)

def horizontal_stripe_dither(img):
    grayscale = np.array(to_grayscale(img), dtype=np.uint8)
    h, w = grayscale.shape
    stripes = np.zeros_like(grayscale)
    for y in range(h):
        threshold = 128 + 50 * np.sin(y / 5.0)
        stripes[y] = (grayscale[y] > threshold) * 255
    return Image.fromarray(stripes.astype(np.uint8), mode="L")

def apply_all(img, output_dir="supa_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        # "no_dither.png": no_dither,
        # "floyd_steinberg.png": floyd_steinberg,
        # "posterize_1bit.png": posterize_1bit,
        # "posterize_2bit.png": posterize_2bit,
        "bayer_2x2.png": lambda i: bayer_dither(i, np.array([[0,2],[3,1]]) / 4.0),
        "bayer_4x4.png": lambda i: bayer_dither(i, np.array([
            [0,8,2,10],
            [12,4,14,6],
            [3,11,1,9],
            [15,7,13,5]
        ]) / 16.0),
        "bayer_8x8.png": lambda i: bayer_dither(i, (np.kron(np.array([
            [0, 32], [48, 16]
        ]), np.ones((4,4))) / 64.0)),
        # "random_pixel.png": random_dither_pixel,
        # "random_blocky.png": random_dither_blocky,
        # "atkinson_full.png": lambda i: atkinson(i, reduced=False),
        # "atkinson_reduced.png": lambda i: atkinson(i, reduced=True),
        # "custom_3x3.png": custom_3x3_dither,
        # "horizontal_stripe.png": horizontal_stripe_dither,
        # "halftone_12px.png": lambda i: halftone_dither(i, block_size=12)

    }

    for name, method in methods.items():
        print(f"⏳ Processing {name}...")
        result = method(img)
        result.save(os.path.join(output_dir, name))

    print(f"\n;) All dithers complete. Images saved in '{output_dir}'.")

if __name__ == "__main__":
    filename = input("Enter image filename (in same folder): ").strip()
    try:
        img = load_image(filename)
        apply_all(img)
    except Exception as e:
        print(f":( Error: {e}")
