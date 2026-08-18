"""Immutable geometry primitives shared by every OS-layer module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def offset(self, dx: int, dy: int) -> Point:
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in absolute screen coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> Point:
        return Point(self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, point: Point) -> bool:
        return self.x <= point.x < self.right and self.y <= point.y < self.bottom

    def intersects(self, other: Rect) -> bool:
        return not (
            other.x >= self.right
            or other.right <= self.x
            or other.y >= self.bottom
            or other.bottom <= self.y
        )

    def intersection(self, other: Rect) -> Rect:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        return Rect(x, y, max(0, right - x), max(0, bottom - y))

    def clamp(self, point: Point) -> Point:
        return Point(
            min(max(point.x, self.x), self.right - 1),
            min(max(point.y, self.y), self.bottom - 1),
        )

    def scaled(self, factor: float) -> Rect:
        return Rect(
            round(self.x * factor),
            round(self.y * factor),
            round(self.width * factor),
            round(self.height * factor),
        )

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict) -> Rect:
        return cls(int(data["x"]), int(data["y"]), int(data["width"]), int(data["height"]))


@dataclass(frozen=True, slots=True)
class Monitor:
    """A physical output plus its position in the virtual screen."""

    name: str
    bounds: Rect
    primary: bool = False

    def to_dict(self) -> dict:
        return {"name": self.name, "primary": self.primary, **self.bounds.to_dict()}
