from __future__ import annotations

from enum import Enum
from math import ceil
from pathlib import Path
from typing import Annotated, Any, List, Literal, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from src.utils.infra.image_utils import resize_image
from src.utils.PIL_utils import (
    TRANSPARENT_WHITE,
    LegendItem,
    PrefixStyle,
    TAnchor,
    _get_text_height,
    _safe_font,
    bold_dejavu_path,
    dejavu_path,
    generate_prefix,
)
from src.utils.streamlit.components.extended_streamlit_pydantic import (
    annotate_dict_with_literal_values,
)
from src.utils.streamlit.st_pydantic_v2.input import SpecialFieldKeys
from src.utils.streamlit.ui_pydantic_v2.extra_types import PercentageCrop


class GridOrganizer(BaseModel):
    """Organizes images into a grid based on row and column categories.

    This version doesn't use callable fields to avoid Pydantic serialization issues.
    """

    row_order: Optional[List[Union[str, int, Enum]]] = Field(
        default=None, description="Custom ordering for rows (optional)"
    )
    col_order: Optional[List[Union[str, int, Enum]]] = Field(
        default=None, description="Custom ordering for columns (optional)"
    )


def organize_images_to_grid_with_keys(
    items_with_keys: List[Tuple[Path, Any, Any]],  # (path, row_key, col_key)
    organizer: GridOrganizer,
) -> List[List[Optional[Path]]]:
    """Organize image paths into a grid based on pre-extracted row and column keys.

    Args:
        items_with_keys: List of tuples containing (image_path, row_key, col_key)
        organizer: Configuration for how to organize the grid

    Returns:
        A 2D grid (list of lists) of image paths organized by row and column categories
    """
    # Extract unique row and column keys
    row_keys = set()
    col_keys = set()

    for _, row_key, col_key in items_with_keys:
        row_keys.add(row_key)
        col_keys.add(col_key)

    # Use provided order or sort naturally
    if organizer.row_order:
        sorted_row_keys = [key for key in organizer.row_order if key in row_keys]
    else:
        sorted_row_keys = sorted(row_keys)

    if organizer.col_order:
        sorted_col_keys = [key for key in organizer.col_order if key in col_keys]
    else:
        sorted_col_keys = sorted(col_keys)

    # Create mapping from keys to indices
    row_indices = {key: idx for idx, key in enumerate(sorted_row_keys)}
    col_indices = {key: idx for idx, key in enumerate(sorted_col_keys)}

    # Initialize empty grid
    num_rows = len(sorted_row_keys)
    num_cols = len(sorted_col_keys)
    grid: List[List[Optional[Path]]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    # Place items in the grid
    for img_path, row_key, col_key in items_with_keys:
        if row_key in row_indices and col_key in col_indices:
            row_idx = row_indices[row_key]
            col_idx = col_indices[col_key]
            grid[row_idx][col_idx] = img_path

    return grid


crop_kwargs = {
    SpecialFieldKeys.kwargs: {
        "min_value": 0.0,
        "max_value": 1.0,
        "step": 0.0001,
        "format": "%.4f",
    }
}


class CropParams(BaseModel):
    """Configuration for image cropping with special handling for edge images.

    All values are between 0.0 and 1.0, representing percentages of the image dimensions.
    Edge parameters apply to ALL images as a base crop amount.
    Regular crop parameters are ADDITIONAL crop amounts applied only to non-edge images.
    """

    enable_crop: bool = Field(default=True, description="Enable image cropping")

    # Replace numerical fields with Crop objects
    edge_crop: PercentageCrop = Field(
        default_factory=lambda: PercentageCrop(
            left=3.43,
            top=20.5,
            width=96.3,
            height=79.5,
        ),
        description="Base crop for all images",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_edge"},
    )

    standard_crop: PercentageCrop = Field(
        default_factory=lambda: PercentageCrop(
            left=14.57,
            top=19.29,
            width=84.72,
            height=69.5,
        ),
        description="Additional crop for non-edge images",
        json_schema_extra={SpecialFieldKeys.column_group: "crop_standard"},
    )

    @classmethod
    def set_image(cls, image: Image.Image):
        pass

        def _attach(fi: FieldInfo):
            """
            Clone `fi`, merge the extra JSON‑schema metadata, and
            return an Annotated type that Pydantic will pick up.
            """
            # Create a new dictionary with the desired values
            json_schema_extra = {}
            if fi.json_schema_extra is not None:
                # Copy each key-value pair manually
                if callable(fi.json_schema_extra):
                    fi.json_schema_extra(json_schema_extra)
                else:
                    for key, value in fi.json_schema_extra.items():
                        json_schema_extra[key] = value

            # Add new key-value pairs
            json_schema_extra["image"] = image
            json_schema_extra["aspect_dict"] = "Free"

            return Annotated[
                PercentageCrop,
                fi.merge_field_infos(json_schema_extra=json_schema_extra),
            ]

        # `create_model` gives us a fresh class that inherits every validator,
        # config option, etc. from `cls`; we simply override the two fields.
        return create_model(  # type: ignore[misc]
            f"{cls.__name__}WithImage",
            __base__=cls,
            edge_crop=_attach(cls.model_fields["edge_crop"]),
            standard_crop=_attach(cls.model_fields["standard_crop"]),
        )


class LegendParams(BaseModel):
    """Configuration for the legend appearance."""

    width: int = Field(
        default=80,
        ge=0,
        description="Width of color/line sample",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    height: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Height of sample as ratio of legend height",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    rows: int = Field(
        default=1,
        ge=1,
        description="Number of rows in the legend",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    horizontal_spacing: int = Field(
        default=50,
        ge=0,
        description="Spacing between sample and text",
        json_schema_extra={SpecialFieldKeys.column_group: "line"},
    )
    position: Literal["top", "bottom"] = Field(
        default="bottom",
        description="Position of legend items.",
        json_schema_extra={SpecialFieldKeys.column_group: "_position"},
    )
    font_size: int = Field(
        default=50, ge=0, description="Font size for legend", json_schema_extra={SpecialFieldKeys.column_group: "text"}
    )
    anchor: TAnchor = Field(
        default="lm", description="Anchor for legend", json_schema_extra={SpecialFieldKeys.column_group: "text"}
    )
    show_divider: bool = Field(
        default=True,
        title="Show",
        description="Show divider line above legend",
        json_schema_extra={SpecialFieldKeys.column_group: "divider"},
    )
    legend_padding: int = Field(
        default=30,
        ge=0,
        description="Padding around the divider line",
        json_schema_extra={SpecialFieldKeys.column_group: "divider"},
    )
    divider_width: int = Field(
        default=1,
        ge=0,
        description="Width of divider line",
        json_schema_extra={SpecialFieldKeys.column_group: "divider"},
    )


class ImageGridParams(BaseModel):
    title: str = Field(
        default="", description="Title for the grid", json_schema_extra={SpecialFieldKeys.column_group: "titles"}
    )

    title_height: int = Field(
        default=30,
        ge=0,
        description="Height for titles",
        json_schema_extra={SpecialFieldKeys.column_group: "titles"},
    )
    # font_style: Literal[ name of fonts...
    font_size: int = Field(
        default=25, ge=0, description="Font size", json_schema_extra={SpecialFieldKeys.column_group: "titles"}
    )

    padding: int = Field(
        default=0,
        ge=0,
        description="Padding between images",
        json_schema_extra={SpecialFieldKeys.column_group: "img_size"},
    )
    allow_different_image_sizes: bool = Field(
        default=False,
        description="Allow different image sizes",
    )

    # Separator
    sep1: None = Field(default=None, json_schema_extra={SpecialFieldKeys.separator: True})

    column_header_padding: float = Field(
        default=-10.0,
        description="Additional padding for column headers (can be negative)",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    show_row_labels: bool = Field(
        default=False,
        description="Show row labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )
    show_col_labels: bool = Field(
        default=True,
        description="Show column labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    row_prefix_style: PrefixStyle = Field(
        default="none",
        description="Prefix style for row labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    column_prefix_style: PrefixStyle = Field(
        default="lowercase_letter_paren",
        description="Prefix style for column labels",
        json_schema_extra={SpecialFieldKeys.column_group: "labels_display"},
    )

    rows_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for rows")
    columns_labels_override: dict[str, str] = Field(default_factory=dict, description="Override labels for columns")

    label_font_size: int = Field(default=70, ge=0, description="Font size for labels")

    # Put crop params in an expander
    crop_params: CropParams = Field(
        default_factory=CropParams,
        description="Image cropping configuration",
        json_schema_extra={SpecialFieldKeys.expander: "Crop Settings"},
    )

    # Legend configuration
    legend_params: LegendParams = Field(
        default_factory=LegendParams,
        description="Legend appearance configuration",
        json_schema_extra={SpecialFieldKeys.expander: "Legend Settings"},
    )

    # Convenience constructor for static row/column names -------------------- #
    @classmethod
    def specify_config(
        cls, rows: list[str], columns: list[str], image: Optional[Image.Image] = None
    ) -> ImageGridParams:
        overrides = {}
        if not rows:
            overrides["show_row_labels"] = Annotated[Literal[False], Field(default=False)]
        if not columns:
            overrides["show_col_labels"] = Annotated[Literal[False], Field(default=False)]
        if image is not None:
            image_grid_params = CropParams.set_image(image)
            overrides["crop_params"] = Annotated[image_grid_params, Field(default_factory=image_grid_params)]
        return create_model(  # type: ignore
            f"{cls.__name__}Config",
            __base__=cls,
            rows_labels_override=annotate_dict_with_literal_values(rows, str),
            columns_labels_override=annotate_dict_with_literal_values(columns, str),
            **overrides,
        )


# --------------------------------------------------------------------------- #
#  Core logic                                                                 #
# --------------------------------------------------------------------------- #


class ImageGridGenerator:
    def __init__(
        self,
        images_paths_grid: List[List[Path]],
        params: ImageGridParams,
        legend_items: list[LegendItem] | dict[int, list[LegendItem]],
        row_labels: list[str],
        col_labels: list[str],
    ):
        self.images_paths_grid = images_paths_grid
        self.params = params
        self.row_labels = row_labels
        self.col_labels = col_labels

        self.font_title: ImageFont.FreeTypeFont | ImageFont.ImageFont
        self.font_label: ImageFont.FreeTypeFont | ImageFont.ImageFont
        self.font_legend: ImageFont.FreeTypeFont | ImageFont.ImageFont

        self.num_rows: int = len(images_paths_grid)
        self.num_cols: int = len(images_paths_grid[0]) if images_paths_grid else 0

        if isinstance(legend_items, dict):
            self.is_multi_row_legend = True
            self.legend_items_dict: dict[int, list[LegendItem]] = legend_items
        else:
            self.is_multi_row_legend = False
            if self.params.legend_params.position == "top":
                self.legend_items_dict = {0: legend_items}
            else:
                self.legend_items_dict = {self.num_rows - 1: legend_items}

        self.img_w: float = 0.0
        self.img_h: float = 0.0
        self.original_image_size: Tuple[int, int] = (0, 0)
        self.processed_images: list[list[Optional[Image.Image]]] = []

        self.canvas_w: int = 0
        self.canvas_h: int = 0
        self.top_margin: float = 0.0
        self.bottom_margin: float = 0.0
        self.left_margin: float = 0.0
        self.right_margin: float = 0.0
        self.row_legend_h: float = 0.0
        self.crop_extras: list[float] = [0.0] * 4  # ltrb

        self.canvas: Optional[Image.Image] = None
        self.draw: Optional[ImageDraw.ImageDraw] = None

    def _initialize_fonts(self) -> None:
        self.font_title = _safe_font(bold_dejavu_path, self.params.font_size)
        self.font_label = _safe_font(dejavu_path, self.params.label_font_size)
        self.font_legend = _safe_font(dejavu_path, self.params.legend_params.font_size)

    def _calculate_dimensions(self) -> None:
        # Title row (optional) + column-label row (optional)
        title_h = self.params.title_height if self.params.title else 0
        col_label_h = _get_text_height("TEST", self.font_label) if self.params.show_col_labels else 0
        padded_col_label_h = max(0, col_label_h + self.params.column_header_padding)

        legend_line_h = _get_text_height("TEST", self.font_legend) if self.legend_items_dict else 0

        self.row_legend_h = 0
        if self.legend_items_dict:
            self.row_legend_h = (
                legend_line_h * self.params.legend_params.rows + self.params.legend_params.legend_padding
            )
            if self.params.legend_params.show_divider:  # This is for a single legend block's internal structure
                self.row_legend_h += self.params.legend_params.divider_width

        row_label_w = (
            _get_text_height("TEST", self.font_label) if self.params.show_row_labels else 0
        )  # Width of row label is text height

        # Initialize margins
        self.top_margin = title_h + padded_col_label_h
        self.bottom_margin = 0
        self.left_margin = row_label_w
        self.right_margin = 0

        standard_crop = self.params.crop_params.standard_crop.box.to_unit("fraction")
        edge_crop = self.params.crop_params.edge_crop.box.to_unit("fraction")

        # Temporarily load the first image to get its size for calculations
        # This part will be re-done in _load_and_process_images, but needed here for dimensions
        first_image_path = None
        for row in self.images_paths_grid:
            for img_path in row:
                if img_path:
                    first_image_path = img_path
                    break
            if first_image_path:
                break

        assert first_image_path is not None, "No images found"
        with Image.open(first_image_path) as im_temp:
            im_temp = resize_image(im_temp)
            self.original_image_size = im_temp.size

        if self.params.crop_params.enable_crop:
            self.crop_extras = list(
                edge_crop.gap(standard_crop).to_unit(to_unit_type="absolute", dimensions=self.original_image_size).ltrb
            )
            self.img_w = standard_crop.width * self.original_image_size[0]
            self.img_h = standard_crop.height * self.original_image_size[1]
            self.left_margin += self.crop_extras[0]
            self.right_margin += self.crop_extras[2]
            self.top_margin += self.crop_extras[1]
            self.bottom_margin += self.crop_extras[3]
        else:
            self.crop_extras = [0.0] * 4
            self.img_w, self.img_h = self.original_image_size

        grid_w = self.num_cols * self.img_w + max(self.num_cols - 1, 0) * self.params.padding
        grid_h = self.num_rows * self.img_h + max(self.num_rows - 1, 0) * self.params.padding

        self.canvas_w = int(self.left_margin + grid_w + self.right_margin)
        # Total legend height considers it can appear after multiple rows
        self.canvas_h = int(
            self.top_margin
            + grid_h
            + self.bottom_margin
            + self.row_legend_h * len(self.legend_items_dict)  # Total height of all legends
        )

    def _load_and_process_images(self) -> None:
        self.processed_images = []
        standard_crop = self.params.crop_params.standard_crop.box.to_unit("fraction")
        edge_crop = self.params.crop_params.edge_crop.box.to_unit("fraction")

        first_image_loaded = False
        for i, row_images_paths in enumerate(self.images_paths_grid):
            self.processed_images.append([])
            row_images_list = self.processed_images[-1]
            for j, img_path in enumerate(row_images_paths):
                if img_path is None:
                    row_images_list.append(None)
                    continue
                with Image.open(img_path) as im:
                    im = resize_image(im)
                    if not first_image_loaded:
                        self.original_image_size = im.size  # Update with actual first image
                        first_image_loaded = True
                        # Recalculate dimensions if first image size was a placeholder or differs
                        # This is a simplified approach; a more robust one might recalculate everything
                        # or ensure all images are pre-checked for size if strict sizing is needed.
                        if self.params.crop_params.enable_crop:  # Re-calc img_w, img_h based on actual first image
                            self.crop_extras = list(
                                edge_crop.gap(standard_crop)
                                .to_unit(to_unit_type="absolute", dimensions=self.original_image_size)
                                .ltrb
                            )
                            self.img_w = standard_crop.width * self.original_image_size[0]
                            self.img_h = standard_crop.height * self.original_image_size[1]
                        else:
                            self.img_w, self.img_h = self.original_image_size

                    elif not self.params.allow_different_image_sizes:
                        assert im.size == self.original_image_size, (
                            f"Image {img_path} has a different size {im.size}"
                            f" than the first image {self.original_image_size}"
                        )

                    if self.params.crop_params.enable_crop:
                        crop_box = standard_crop.set_by_mask(
                            edge_crop,
                            {
                                "left": j == 0,
                                "right": j == self.num_cols - 1,
                                "top": i == 0,
                                "bottom": i == self.num_rows - 1,
                            },
                        ).to_unit(to_unit_type="absolute", dimensions=im.size)
                        im = im.crop(crop_box.ltrb)
                    row_images_list.append(im)
        # After all images processed and final original_image_size is known,
        # ensure dependent dimensions are final.
        # This is important if the first image path check in _calculate_dimensions
        # used a placeholder or if allow_different_image_sizes is true and original_image_size may change.
        self._calculate_dimensions()  # Re-run with potentially updated self.original_image_size

    def _create_canvas(self) -> None:
        assert self.canvas_w > 0 and self.canvas_h > 0, "Canvas dimensions must be positive."
        assert self.canvas_w + self.canvas_h < 2e4, (
            f"Canvas size is too large: {self.canvas_w} + {self.canvas_h} = {self.canvas_w + self.canvas_h}"
        )  # Increased limit slightly
        self.canvas = Image.new("RGBA", (self.canvas_w, self.canvas_h), "white")
        self.draw = ImageDraw.Draw(self.canvas)

    def _draw_title(self) -> None:
        if self.params.title and self.draw and self.canvas:
            bb = self.draw.textbbox((0, 0), self.params.title, font=self.font_title)
            txt_w, txt_h = bb[2] - bb[0], bb[3] - bb[1]

            self.draw.text(
                ((self.canvas_w - txt_w) // 2, (self.params.title_height - txt_h) // 2),
                self.params.title,
                fill="black",
                font=self.font_title,
            )

    def _draw_row_labels_and_images(self) -> None:
        if not self.draw or not self.canvas:
            return

        if self.params.show_row_labels:
            assert len(self.row_labels) == self.num_rows, "Number of row labels must match number of rows."

        cum_y_top = self.top_margin

        for row_idx, row_images_list in enumerate(self.processed_images):
            current_row_start_y = cum_y_top  # Y where this entire row's content block begins
            y_offset_in_row = 0.0  # Accumulated height within this row (e.g., for a top legend)

            # Draw legend for this row if it's positioned at the top AND IS MULTI-ROW
            if row_idx in self.legend_items_dict and self.params.legend_params.position == "top":
                self._draw_one_legend_set(
                    legend_items_list=self.legend_items_dict[row_idx],
                    legend_y_start=current_row_start_y,  # Multi-row top legend starts at current row's top
                    current_row_idx=row_idx,
                )
                y_offset_in_row += self.row_legend_h

            y_for_images_in_row = current_row_start_y + y_offset_in_row

            # Row label (once per row)
            if self.params.show_row_labels:
                label_text = self.row_labels[row_idx]
                if label_text in self.params.rows_labels_override:
                    label_text = self.params.rows_labels_override[label_text]

                prefix = generate_prefix(row_idx, self.params.row_prefix_style)
                if prefix:
                    label_text = f"{prefix} {label_text}"

                actual_img_h_for_label = self.img_h
                for img in row_images_list:
                    if img:
                        actual_img_h_for_label = img.height
                        break

                label_canvas_w = actual_img_h_for_label
                label_text_h = _get_text_height(label_text, self.font_label)
                label_img = Image.new("RGBA", (int(label_canvas_w), int(label_text_h)), TRANSPARENT_WHITE)
                label_draw_obj = ImageDraw.Draw(label_img)
                bb = label_draw_obj.textbbox((0, 0), label_text, font=self.font_label)
                txt_w_for_label = bb[2] - bb[0]
                label_draw_obj.text(
                    ((label_canvas_w - txt_w_for_label) / 2, 0),
                    label_text,
                    fill="black",
                    font=self.font_label,
                )
                label_img = label_img.rotate(90, expand=True)
                self.canvas.paste(label_img, (0, int(y_for_images_in_row)), label_img)

            # Images
            current_max_img_h_in_row = 0
            for col_idx, img in enumerate(row_images_list):
                x_left = self.left_margin + col_idx * (self.img_w + self.params.padding)
                if col_idx == 0:
                    x_left -= self.crop_extras[0]
                if img is None:
                    continue
                self.canvas.paste(img, (int(x_left), int(y_for_images_in_row)))
                current_max_img_h_in_row = max(current_max_img_h_in_row, img.height)

            effective_row_height = current_max_img_h_in_row if current_max_img_h_in_row > 0 else self.img_h

            cum_y_top = current_row_start_y + y_offset_in_row + effective_row_height + self.params.padding

            if row_idx in self.legend_items_dict and self.params.legend_params.position == "bottom":
                effective_row_idx_for_divider_bottom = row_idx
                if not self.is_multi_row_legend:
                    if row_idx == (self.num_rows - 1) or self.num_rows == 1:
                        effective_row_idx_for_divider_bottom = self.num_rows - 1
                self._draw_one_legend_set(
                    legend_items_list=self.legend_items_dict[row_idx],
                    legend_y_start=cum_y_top,
                    current_row_idx=effective_row_idx_for_divider_bottom,
                )
                cum_y_top += self.row_legend_h

    def _draw_col_labels(self) -> None:
        if not self.params.show_col_labels or not self.draw or not self.canvas:
            return

        assert len(self.col_labels) == self.num_cols, "Number of col labels must match number of columns."

        col_labels_y_start = 0.0
        if self.params.title:
            col_labels_y_start += self.params.title_height

        col_label_text_h = _get_text_height("TEST", self.font_label)
        for col_idx in range(self.num_cols):
            prefix = generate_prefix(col_idx, self.params.column_prefix_style)
            label_text = self.col_labels[col_idx]
            if label_text in self.params.columns_labels_override:
                label_text = self.params.columns_labels_override[label_text]
            if prefix:
                label_text = f"{prefix} {label_text}"

            label_img_w = self.img_w
            label_img_h = col_label_text_h
            label_img = Image.new("RGBA", (int(label_img_w), int(label_img_h)), TRANSPARENT_WHITE)
            label_draw_obj = ImageDraw.Draw(label_img)
            bb = label_draw_obj.textbbox((0, 0), label_text, font=self.font_label)
            txt_w_for_label = bb[2] - bb[0]
            text_y_on_label_canvas = 0
            label_draw_obj.text(
                ((label_img_w - txt_w_for_label) / 2, text_y_on_label_canvas),
                label_text,
                fill="black",
                font=self.font_label,
            )
            paste_x = self.left_margin + col_idx * (self.img_w + self.params.padding)
            paste_y = col_labels_y_start
            if self.params.column_header_padding >= 0:
                paste_y += self.params.column_header_padding
            self.canvas.paste(label_img, (int(paste_x), int(paste_y)), label_img)

    def _draw_one_legend_set(
        self, legend_items_list: list[LegendItem], legend_y_start: float, current_row_idx: int
    ) -> None:
        if not legend_items_list or not self.draw or not self.canvas:
            return

        legend_padding_half = self.params.legend_params.legend_padding / 2
        divider_h = self.params.legend_params.divider_width

        show_this_divider = self.params.legend_params.show_divider
        if self.is_multi_row_legend:
            if self.params.legend_params.position == "top" and current_row_idx == 0:
                show_this_divider = False
            elif self.params.legend_params.position == "bottom" and current_row_idx == self.num_rows - 1:
                show_this_divider = False
        # If params.legend_params.show_divider is False initially, show_this_divider remains False.

        # Determine the physical placement of the divider (top or bottom of its block)
        if self.is_multi_row_legend:
            divider_is_physically_at_top_of_block = self.params.legend_params.position == "top"
        else:  # Global legend
            divider_is_physically_at_top_of_block = self.params.legend_params.position == "bottom"

        # y_base_for_legend_content is where drawing starts after the block's overall top padding
        y_base_for_legend_content = legend_y_start + legend_padding_half
        items_loop_start_y = (
            y_base_for_legend_content  # This is where items rendering begins, possibly adjusted by a top divider
        )

        if divider_is_physically_at_top_of_block:
            if show_this_divider:
                # Draw divider at the top of the content area (after top padding, before items)
                divider_y_top = y_base_for_legend_content
                self.draw.line(
                    [(0, divider_y_top), (self.canvas_w, divider_y_top)],
                    fill="black",
                    width=divider_h,
                )
                items_loop_start_y += divider_h  # Items start below this divider
        # else: Divider is at bottom or not shown; items_loop_start_y is already set after top padding.

        num_items = len(legend_items_list)
        actual_rows = min(self.params.legend_params.rows, num_items)
        items_per_row = ceil(num_items / actual_rows)

        available_legend_content_h = (
            self.row_legend_h - self.params.legend_params.legend_padding  # Total padding (top+bottom)
        )
        if show_this_divider:  # If a divider is actually shown (either top or bottom), it consumes space
            available_legend_content_h -= divider_h

        row_content_height = available_legend_content_h / actual_rows if actual_rows > 0 else 0

        sample_width = self.params.legend_params.width
        sample_to_text_gap = 10
        inter_item_spacing = self.params.legend_params.horizontal_spacing

        item_index = 0
        for legend_row_idx in range(actual_rows):  # Renamed row_idx to legend_row_idx to avoid clash
            row_items = legend_items_list[item_index : item_index + items_per_row]
            if not row_items:
                continue
            item_index += items_per_row

            per_item_widths: list[float] = []
            for item in row_items:
                bb = self.draw.textbbox((0, 0), item.label, font=self.font_legend)
                txt_w = bb[2] - bb[0]
                per_item_widths.append(sample_width + sample_to_text_gap + txt_w)

            if not per_item_widths:
                continue

            row_total_w = sum(per_item_widths) + inter_item_spacing * (len(per_item_widths) - 1)
            cur_x = (self.canvas_w - row_total_w) / 2

            # Y position for the content of this specific legend row, relative to items_loop_start_y
            y_row_content_start = items_loop_start_y + (legend_row_idx * row_content_height)

            for item, item_w in zip(row_items, per_item_widths):
                sample_h_ratio = self.params.legend_params.height
                actual_sample_h = row_content_height * sample_h_ratio
                sample_y_center = y_row_content_start + (row_content_height / 2)
                line_y = sample_y_center

                if item.linestyle in ("--", ":"):
                    dash_len, gap_len = (6, 3) if item.linestyle == "--" else (2, 6)
                    pos, end_x = cur_x, cur_x + sample_width
                    while pos < end_x:
                        self.draw.line(
                            [(pos, line_y), (min(pos + dash_len, end_x), line_y)],
                            fill=item.color,
                            width=int(actual_sample_h * 0.2),
                        )
                        pos += dash_len + gap_len
                else:
                    self.draw.line(
                        [(cur_x, line_y), (cur_x + sample_width, line_y)],
                        fill=item.color,
                        width=int(actual_sample_h * 0.4),
                    )

                text_x = cur_x + sample_width + sample_to_text_gap
                text_y_anchor_point = y_row_content_start + row_content_height / 2
                self.draw.text(
                    (text_x, text_y_anchor_point),
                    item.label,
                    fill="black",
                    font=self.font_legend,
                    anchor=self.params.legend_params.anchor,
                )
                cur_x += item_w + inter_item_spacing

        # Draw divider at the bottom if needed
        if not divider_is_physically_at_top_of_block and show_this_divider:
            # Divider is at the bottom of the content area, just above the overall bottom padding of the legend block.
            divider_y_bottom = legend_y_start + self.row_legend_h - legend_padding_half - divider_h
            self.draw.line(
                [(0, divider_y_bottom), (self.canvas_w, divider_y_bottom)],
                fill="black",
                width=divider_h,
            )

    def generate(self) -> Image.Image:
        if not self.images_paths_grid or not self.images_paths_grid[0]:
            # Handle empty grid case: return a small blank image or raise error
            print("Warning: Empty image grid provided.")
            empty_canvas = Image.new("RGBA", (100, 50), "white")
            draw_empty = ImageDraw.Draw(empty_canvas)
            draw_empty.text((10, 10), "Empty Grid", fill="black")
            return empty_canvas

        self._initialize_fonts()
        # Initial dimension calculation (may be refined after first image load)
        self._calculate_dimensions()
        # Load images and finalize dimensions based on actual image sizes
        # _load_and_process_images now calls _calculate_dimensions() at its end.
        self._load_and_process_images()
        # Create canvas with final dimensions
        self._create_canvas()

        if not self.canvas or not self.draw:
            raise RuntimeError("Canvas or Draw context not initialized.")

        # --- Start Drawing ---
        y_cursor = 0.0  # Keeps track of vertical position for drawing top elements

        if self.params.title:
            self._draw_title()  # Draws title near y=0 within its allocated title_height
            y_cursor = float(self.params.title_height)

        if self.params.show_col_labels:
            # _draw_col_labels internally calculates its y based on title_height.
            # It effectively starts drawing from the current y_cursor.
            self._draw_col_labels()
            # Update y_cursor to be after column labels region
            col_label_text_h = _get_text_height("TEST", self.font_label)
            # This is the space reserved by _calculate_dimensions for column labels region
            actual_col_label_region_h = max(0, col_label_text_h + self.params.column_header_padding)
            y_cursor += actual_col_label_region_h

        self._draw_row_labels_and_images()

        return self.canvas


def combine_image_grid(
    images_paths_grid: List[List[Path]],
    params: ImageGridParams,
    legend_items: list[LegendItem] | dict[int, list[LegendItem]],
    row_labels: list[str],
    col_labels: list[str],
) -> Image.Image:
    generator = ImageGridGenerator(
        images_paths_grid=images_paths_grid,
        params=params,
        legend_items=legend_items,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    return generator.generate()
