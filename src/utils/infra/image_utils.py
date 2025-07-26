from PIL import Image

ACL_ROW_SIZE_IN = 6.3
IMAGE_SIZE_IN = 1.5
DPI = 600


def resize_image(image: Image.Image, target_width_in: float = IMAGE_SIZE_IN, dpi: int = DPI):
    # 1. Compute needed pixel width
    target_px = int(target_width_in * dpi)
    # 2. Compute the same scale factor for height
    scale = target_px / image.width
    target_py = int(image.height * scale)
    return image.resize((target_px, target_py), resample=Image.Resampling.LANCZOS)


def save_at_dpi(image: Image.Image, out_path: str, target_width_in: float = ACL_ROW_SIZE_IN, dpi: int = DPI):
    resized = resize_image(image, target_width_in, dpi)
    resized.save(out_path, dpi=(dpi, dpi))
