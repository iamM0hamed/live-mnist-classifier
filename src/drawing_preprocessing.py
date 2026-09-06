import numpy as np
from PIL import Image


def to_grayscale(image):
    image = Image.fromarray(image)
    image = image.convert("L")
    return np.array(image)


def get_digit_mask(image):
    """
    return pixels in boolen where pixel value = True
    if its value > mask value
    """
    return image > 20  # mask = 20


def get_digit_coordinates(mask):
    """
    return pixels coordinates where pixel value = True
    """
    return np.argwhere(mask)


def get_bounding_box(image):
    mask = get_digit_mask(image)
    coords = get_digit_coordinates(mask)

    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return y_min, x_min, y_max, x_max


def crop_digit(image, bounding_box):
    y_min, x_min, y_max, x_max = get_bounding_box(image)
    return image[y_min : y_max + 1, x_min : x_max + 1]


def resize_digit(image, target_size=20):
    height, width = image.shape
    scale = target_size / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    image = Image.fromarray(image)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(image)


def center_digit(image, canvas_size=28):
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    height, width = image.shape

    y_offset = (canvas_size - height) // 2
    x_offset = (canvas_size - width) // 2

    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = image
    return canvas


def prepare_for_model(image):
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_drawing(image):
    grayscale = to_grayscale(image)
    bounding_box = get_bounding_box(grayscale)
    if bounding_box is None:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)
    cropped = crop_digit(grayscale, bounding_box)
    resized = resize_digit(cropped)
    centered = center_digit(resized)

    return prepare_for_model(centered)


if __name__ == "__main__":
    test = np.zeros((560, 560, 4), dtype=np.uint8)

    result = preprocess_drawing(test)

    print("Final shape:", result.shape)
    print("Final dtype:", result.dtype)
    print("Final min:", result.min())
    print("Final max:", result.max())
