from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, cast

from pydantic import BaseModel, Field, model_validator

T_XYWH = Literal["left", "top", "width", "height"]
T_LTRB = Literal["left", "top", "right", "bottom"]

XYWH_SIDES = cast(tuple[T_XYWH, ...], ("left", "top", "width", "height"))
LTRB_SIDES = cast(tuple[T_LTRB, ...], ("left", "top", "right", "bottom"))


ALL_SIDES = T_XYWH | T_LTRB

UnitType = Literal["absolute", "percentage", "fraction"]


@dataclass(frozen=True)
class Box:
    number_type: UnitType
    left: float
    top: float
    right: float
    bottom: float
    ndigits: int = 2

    def __post_init__(self):
        for side in LTRB_SIDES:
            object.__setattr__(self, side, round(getattr(self, side), self.ndigits))

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @classmethod
    def empty(cls) -> Box:
        return cls(number_type="absolute", left=0, top=0, right=0, bottom=0)

    @classmethod
    def from_xywh(cls, number_type: UnitType, left: float, top: float, width: float, height: float) -> Box:
        return cls(number_type=number_type, left=left, top=top, right=left + width, bottom=top + height)

    def to_unit(
        self,
        to_unit_type: UnitType,
        dimensions: Optional[tuple[float, float]] = None,
    ) -> Box:
        if self.number_type == to_unit_type:
            return self

        total_w = total_h = 0
        # If either current or target is absolute we must have dimensions
        if "absolute" in (self.number_type, to_unit_type):
            assert dimensions is not None, "dimensions must be provided when converting to or from 'absolute'"
            total_w, total_h = dimensions

        # Step 1: convert self to fraction (0.0–1.0)
        if self.number_type == "absolute":
            left = self.left / total_w
            top = self.top / total_h
            right = self.right / total_w
            bottom = self.bottom / total_h
        elif self.number_type == "percentage":
            left = self.left / 100.0
            top = self.top / 100.0
            right = self.right / 100.0
            bottom = self.bottom / 100.0
        else:  # already fraction
            left, top, right, bottom = self.left, self.top, self.right, self.bottom

        # Step 2: scale fraction into target units
        if to_unit_type == "fraction":
            new_vals = (left, top, right, bottom)
        elif to_unit_type == "percentage":
            new_vals = (left * 100.0, top * 100.0, right * 100.0, bottom * 100.0)
        else:  # to_unit_type == "absolute"
            new_vals = (left * total_w, top * total_h, right * total_w, bottom * total_h)

        return Box(
            number_type=to_unit_type,
            left=new_vals[0],
            top=new_vals[1],
            right=new_vals[2],
            bottom=new_vals[3],
            ndigits=self.ndigits,
        )

    @property
    def ltrb(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.width, self.height)

    @property
    def lrtb(self) -> tuple[float, float, float, float]:
        return (self.left, self.right, self.top, self.bottom)

    def gap(self, other: Box) -> Box:
        assert isinstance(other, Box)
        assert self.number_type == other.number_type
        assert self.ndigits == other.ndigits
        return replace(
            self,
            left=other.left - self.left,
            top=other.top - self.top,
            right=self.right - other.right,
            bottom=self.bottom - other.bottom,
        )

    def get_side(self, side: ALL_SIDES) -> float:
        return getattr(self, side)

    def add_to_side(self, side: T_LTRB, value: float) -> Box:
        return replace(self, **{side: getattr(self, side) + value})

    def multiply_side(self, side: T_LTRB, value: float) -> Box:
        return replace(self, **{side: getattr(self, side) * value})

    def set_by_mask(self, other: Box, mask: dict[T_LTRB, bool]) -> Box:
        new_vals = {}
        for side, should_copy in mask.items():
            if should_copy:
                new_vals[side] = other.get_side(side)

        return replace(self, **new_vals)


class PercentageCrop(BaseModel):
    """Represents crop coordinates as normalized values (0.0-1.0)."""

    left: float = Field(default=0.0, ge=0.0, le=100.0, description="Crop from left (0.0-100.0)")
    top: float = Field(default=0.0, ge=0.0, le=100.0, description="Crop from top (0.0-100.0)")
    width: float = Field(default=90.0, ge=0.0, le=100.0, description="Crop from right (0.0-100.0)")
    height: float = Field(default=90.0, ge=0.0, le=100.0, description="Crop from bottom (0.0-100.0)")

    @model_validator(mode="after")
    def _validate_not_too_small(self):
        if self.width < 5.0:
            self.width = 90.0
        if self.height < 5.0:
            self.height = 90.0

        self.width = min(self.width, 100.0 - self.left)
        self.height = min(self.height, 100.0 - self.top)

        return self

    @property
    def box(self) -> Box:
        return Box.from_xywh(
            number_type="percentage",
            **self.model_dump(),
        )
