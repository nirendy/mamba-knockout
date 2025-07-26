from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# Font Constants
dejavu_path = font_manager.findfont("DejaVu Sans")
bold_dejavu_path = font_manager.findfont(font_manager.FontProperties("DejaVu Sans", weight="bold"))

# Color Constants
TRANSPARENT_WHITE = (255, 255, 255, 0)


# Prefix styles for row and column labels
PrefixStyle = Literal[
    "none",  # No prefix
    "lowercase_letter_paren",  # (a), (b), ...
    "uppercase_letter_paren",  # (A), (B), ...
    "number_paren",  # (1), (2), ...
    "lowercase_roman_paren",  # (i), (ii), ...
    "uppercase_roman_paren",  # (I), (II), ...
    "lowercase_letter_dot",  # a., b., ...
    "uppercase_letter_dot",  # A., B., ...
    "number_dot",  # 1., 2., ...
]


def to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def generate_prefix(index: int, style: PrefixStyle) -> str:
    """Generate a prefix based on the index (0-based) and style."""
    if style == "none":
        return ""

    # Handle letter styles
    if style.startswith("lowercase_letter"):
        char = chr(97 + index)  # 'a' starts at ASCII 97
    elif style.startswith("uppercase_letter"):
        char = chr(65 + index)  # 'A' starts at ASCII 65

    # Handle number styles
    elif style.startswith("number"):
        char = str(index + 1)  # 1-based numbering for human readability

    # Handle roman numeral styles
    elif style.startswith("lowercase_roman"):
        char = to_roman(index + 1).lower()
    elif style.startswith("uppercase_roman"):
        char = to_roman(index + 1)
    else:
        return ""

    # Format with parentheses or dot
    if style.endswith("_paren"):
        return f"({char})"
    elif style.endswith("_dot"):
        return f"{char}."

    return char


def _safe_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Attempt to load the requested font; gracefully fall back to Pillow default."""
    # Fallback to Pillow's default if font loading fails
    try:
        return ImageFont.truetype(path, size)
    except IOError:
        return ImageFont.load_default()


def _get_text_height(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    return ceil(bbox[3] - bbox[1]) * 2


@dataclass
class LegendItem:
    label: str
    color: str
    linestyle: str


TAnchor = Literal[
    "la", "lt", "lm", "ls", "lb", "ld", "ma", "mt", "mm", "ms", "mb", "md", "ra", "rt", "rm", "rs", "rb", "rd"
]
