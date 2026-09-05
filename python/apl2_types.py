"""Core APL2 value types for the Python library."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from decimal import Decimal
from functools import reduce
from operator import mul
from typing import Any, Iterable, Iterator, List, Sequence, Tuple

from apl2_context import current_context


def shape_product(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    return reduce(mul, shape, 1)


def iter_indices(shape: Sequence[int]) -> Iterator[Tuple[int, ...]]:
    if not shape:
        yield ()
        return
    for index in range(shape_product(shape)):
        remainder = index
        coordinates: List[int] = []
        for size in reversed(shape):
            coordinates.append(remainder % size)
            remainder //= size
        yield tuple(reversed(coordinates))


def linear_index(shape: Sequence[int], coordinates: Sequence[int]) -> int:
    if len(shape) != len(coordinates):
        raise IndexError("Coordinate rank does not match array rank")
    index = 0
    for size, coordinate in zip(shape, coordinates):
        if coordinate < 0 or coordinate >= size:
            raise IndexError("Array index out of bounds")
        index = index * size + coordinate
    return index


def normalize_axis(rank: int, axis: int | None, default_last: bool = True) -> int:
    if rank == 0:
        raise ValueError("Axis operations require an array with rank greater than zero")
    resolved = rank - 1 if axis is None and default_last else 0 if axis is None else axis
    if resolved < 0:
        resolved += rank
    if resolved < 0 or resolved >= rank:
        raise ValueError(f"Axis {axis} is out of bounds for rank {rank}")
    return resolved


class APLType:
    """Base type for all APL values."""

    def to_numeric(self) -> float:
        raise TypeError(f"{self.__class__.__name__} cannot be converted to a numeric value")

    def to_boolean(self) -> bool:
        return bool(self.to_python())

    def to_python(self) -> Any:
        raise NotImplementedError

    def deep_copy(self) -> "APLType":
        return copy.deepcopy(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APLType):
            return False
        return _values_equal(
            self.to_python(),
            other.to_python(),
            current_context().comparison_tolerance,
        )


@dataclass
class BooleanType(APLType):
    value: bool

    def to_numeric(self) -> float:
        return 1.0 if self.value else 0.0

    def to_boolean(self) -> bool:
        return self.value

    def to_python(self) -> bool:
        return self.value


@dataclass
class IntegerType(APLType):
    value: int

    def to_numeric(self) -> float:
        return float(self.value)

    def to_python(self) -> int:
        return self.value


@dataclass
class FloatingPointType(APLType):
    value: float

    def to_numeric(self) -> float:
        return self.value

    def to_python(self) -> float:
        return self.value


class ComplexType(APLType):
    def __init__(self, real: float, imaginary: float = 0.0):
        self.real = real
        self.imaginary = imaginary
        self.value = complex(real, imaginary)

    def to_numeric(self) -> float:
        return abs(self.value)

    def to_boolean(self) -> bool:
        return self.value != 0

    def to_python(self) -> complex:
        return self.value


class CharacterType(APLType):
    def __init__(self, value: str):
        if len(value) != 1:
            raise ValueError("CharacterType requires a single character")
        self.value = value

    def to_python(self) -> str:
        return self.value


@dataclass
class StringType(APLType):
    value: str

    def to_boolean(self) -> bool:
        return bool(self.value)

    def to_python(self) -> str:
        return self.value


class BigIntegerType(IntegerType):
    def __init__(self, value: str | int):
        super().__init__(int(value))


class BigDecimalType(APLType):
    def __init__(self, value: str | float | Decimal):
        self.value = Decimal(str(value))

    def to_numeric(self) -> float:
        return float(self.value)

    def to_boolean(self) -> bool:
        return self.value != 0

    def to_python(self) -> Decimal:
        return self.value


class ArrayType(APLType):
    """Flat array storage with explicit shape metadata."""

    def __init__(self, elements: Iterable[Any], shape: Sequence[int] | None = None):
        self.elements = [as_apl_value(element) for element in elements]
        if shape is None:
            shape = (len(self.elements),)
        self.shape = tuple(int(size) for size in shape)
        if any(size < 0 for size in self.shape):
            raise ValueError("Array shape values must be non-negative")
        if shape_product(self.shape) != len(self.elements):
            raise ValueError("Array shape does not match element count")
        self.rank = len(self.shape)

    def size(self) -> int:
        return len(self.elements)

    def flatten(self) -> "ArrayType":
        return ArrayType([element.deep_copy() for element in self.elements], (len(self.elements),))

    def reshape(self, *new_shape: int) -> "ArrayType":
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        resolved_shape = tuple(int(size) for size in new_shape)
        target_size = shape_product(resolved_shape)
        if target_size == 0:
            return ArrayType([], resolved_shape)
        if not self.elements:
            filler = IntegerType(0)
            reshaped = [filler.deep_copy() for _ in range(target_size)]
        else:
            reshaped = [
                self.elements[index % len(self.elements)].deep_copy()
                for index in range(target_size)
            ]
        return ArrayType(reshaped, resolved_shape)

    def get(self, coordinates: Sequence[int], apply_index_origin: bool = True) -> APLType:
        adjusted = tuple(int(coordinate) for coordinate in coordinates)
        if apply_index_origin:
            origin = current_context().index_origin
            adjusted = tuple(coordinate - origin for coordinate in adjusted)
        return self.elements[linear_index(self.shape, adjusted)]

    def to_python(self) -> Any:
        if self.rank == 0:
            return self.elements[0].to_python()
        if self.rank == 1:
            return [element.to_python() for element in self.elements]
        return _nested_python(self.elements, self.shape)

    def transpose(self, axes: Sequence[int] | None = None) -> "ArrayType":
        if axes is None:
            axes = tuple(reversed(range(self.rank)))
        if sorted(axes) != list(range(self.rank)):
            raise ValueError("Axes must be a permutation of array axes")
        new_shape = tuple(self.shape[axis] for axis in axes)
        result = []
        for new_index in iter_indices(new_shape):
            old_index = [0] * self.rank
            for new_axis, old_axis in enumerate(axes):
                old_index[old_axis] = new_index[new_axis]
            result.append(self.get(tuple(old_index), apply_index_origin=False).deep_copy())
        return ArrayType(result, new_shape)

    def to_boolean(self) -> bool:
        return any(element.to_boolean() for element in self.elements)


def _nested_python(elements: Sequence[APLType], shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return [element.to_python() for element in elements]
    cell_size = shape_product(shape[1:])
    return [
        _nested_python(elements[start:start + cell_size], shape[1:])
        for start in range(0, len(elements), cell_size)
    ]


def _values_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(left_item, right_item, tolerance)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, Decimal):
        left = float(left)
    if isinstance(right, Decimal):
        right = float(right)
    if isinstance(left, complex) or isinstance(right, complex):
        left_complex = complex(left)
        right_complex = complex(right)
        return math.isclose(left_complex.real, right_complex.real, rel_tol=0.0, abs_tol=tolerance) and math.isclose(
            left_complex.imag,
            right_complex.imag,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def as_apl_value(value: Any) -> APLType:
    if isinstance(value, APLType):
        return value
    if isinstance(value, bool):
        return BooleanType(value)
    if isinstance(value, int):
        return IntegerType(value)
    if isinstance(value, float):
        return FloatingPointType(value)
    if isinstance(value, complex):
        return ComplexType(value.real, value.imag)
    if isinstance(value, Decimal):
        return BigDecimalType(value)
    if isinstance(value, str):
        return CharacterType(value) if len(value) == 1 else StringType(value)
    if isinstance(value, (list, tuple)):
        return ArrayType(value)
    raise TypeError(f"Unsupported APL value: {type(value)!r}")
