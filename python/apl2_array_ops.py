"""APL2 array operations for Python."""

from __future__ import annotations

from typing import Callable, Sequence

from apl2_types import APLType, ArrayType, BooleanType, IntegerType, as_apl_value, iter_indices, normalize_axis
from apl2_operations import MathOperations


class ArrayOperations:
    """Array operations and transformations for APL2."""

    @staticmethod
    def reshape(array: ArrayType, *new_shape: int) -> ArrayType:
        return array.reshape(*new_shape)

    @staticmethod
    def flatten(array: ArrayType) -> ArrayType:
        return array.flatten()

    @staticmethod
    def ravel(array: ArrayType) -> ArrayType:
        return array.flatten()

    @staticmethod
    def ravel_catenate(array: ArrayType) -> ArrayType:
        return array.flatten()

    @staticmethod
    def shape(array: ArrayType) -> ArrayType:
        return ArrayType([IntegerType(size) for size in array.shape], (len(array.shape),))

    @staticmethod
    def get_shape(array: ArrayType) -> ArrayType:
        return ArrayOperations.shape(array)

    @staticmethod
    def reverse(array: ArrayType, axis: int | None = None) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        return ArrayOperations._transform_axis(
            array,
            resolved_axis,
            lambda index, size: size - 1 - index,
        )

    @staticmethod
    def rotate(array: ArrayType, n: int, axis: int | None = None) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        axis_size = array.shape[resolved_axis]
        if axis_size == 0:
            return ArrayType([], array.shape)
        return ArrayOperations._transform_axis(
            array,
            resolved_axis,
            lambda index, size: (index - n) % size,
        )

    @staticmethod
    def reverse_first_axis(array: ArrayType) -> ArrayType:
        return ArrayOperations.reverse(array, axis=0)

    @staticmethod
    def rotate_first_axis(array: ArrayType, n: int) -> ArrayType:
        return ArrayOperations.rotate(array, n, axis=0)

    @staticmethod
    def transpose(array: ArrayType) -> ArrayType:
        return array.transpose()

    @staticmethod
    def transpose_axes(array: ArrayType, axes: Sequence[int]) -> ArrayType:
        return array.transpose(axes)

    @staticmethod
    def take(array: ArrayType, n: int, axis: int | None = None) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        target_size = abs(n)
        new_shape = list(array.shape)
        new_shape[resolved_axis] = target_size
        filler = IntegerType(0)
        result = []
        start = 0 if n >= 0 else array.shape[resolved_axis] - target_size
        for new_index in iter_indices(tuple(new_shape)):
            source_index = list(new_index)
            source_index[resolved_axis] = start + new_index[resolved_axis]
            if 0 <= source_index[resolved_axis] < array.shape[resolved_axis]:
                result.append(array.get(tuple(source_index)).deep_copy())
            else:
                result.append(filler.deep_copy())
        return ArrayType(result, tuple(new_shape))

    @staticmethod
    def first(array: ArrayType, axis: int | None = None) -> APLType:
        resolved_axis = normalize_axis(array.rank, axis)
        new_shape = array.shape[:resolved_axis] + array.shape[resolved_axis + 1:]
        result = []
        for new_index in iter_indices(new_shape):
            source_index = new_index[:resolved_axis] + (0,) + new_index[resolved_axis:]
            result.append(array.get(source_index).deep_copy())
        return result[0] if not new_shape else ArrayType(result, new_shape)

    @staticmethod
    def drop(array: ArrayType, n: int, axis: int | None = None) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        axis_size = array.shape[resolved_axis]
        drop_count = min(abs(n), axis_size)
        new_size = axis_size - drop_count
        new_shape = list(array.shape)
        new_shape[resolved_axis] = new_size
        offset = drop_count if n >= 0 else 0
        result = []
        for new_index in iter_indices(tuple(new_shape)):
            source_index = list(new_index)
            source_index[resolved_axis] += offset
            result.append(array.get(tuple(source_index)).deep_copy())
        return ArrayType(result, tuple(new_shape))

    @staticmethod
    def split(array: ArrayType, axis: int | None = None) -> ArrayType:
        return ArrayOperations.drop(array, 1, axis=axis)

    @staticmethod
    def catenate(left: ArrayType, right: ArrayType, axis: int | None = None) -> ArrayType:
        if left.rank != right.rank:
            raise ValueError("Catenate requires arrays with the same rank")
        resolved_axis = normalize_axis(left.rank, axis)
        for idx, (left_size, right_size) in enumerate(zip(left.shape, right.shape)):
            if idx != resolved_axis and left_size != right_size:
                raise ValueError("Catenate requires matching shapes on non-concatenated axes")
        new_shape = list(left.shape)
        new_shape[resolved_axis] += right.shape[resolved_axis]
        result = []
        for new_index in iter_indices(tuple(new_shape)):
            if new_index[resolved_axis] < left.shape[resolved_axis]:
                result.append(left.get(new_index).deep_copy())
            else:
                source_index = list(new_index)
                source_index[resolved_axis] -= left.shape[resolved_axis]
                result.append(right.get(tuple(source_index)).deep_copy())
        return ArrayType(result, tuple(new_shape))

    concatenate = catenate

    @staticmethod
    def count(array: ArrayType) -> int:
        return sum(1 for elem in array.elements if isinstance(elem, APLType) and elem.to_boolean())

    @staticmethod
    def sum(array: ArrayType, axis: int | None = None) -> APLType:
        return ArrayOperations.reduce(MathOperations.add, array, axis=axis)

    @staticmethod
    def product(array: ArrayType, axis: int | None = None) -> APLType:
        return ArrayOperations.reduce(MathOperations.multiply, array, axis=axis)

    @staticmethod
    def maximum(array: ArrayType, axis: int | None = None) -> APLType:
        return ArrayOperations.reduce(MathOperations.max, array, axis=axis)

    @staticmethod
    def minimum(array: ArrayType, axis: int | None = None) -> APLType:
        return ArrayOperations.reduce(MathOperations.min, array, axis=axis)

    @staticmethod
    def reduce(function: Callable[[APLType, APLType], APLType], array: ArrayType, axis: int | None = None) -> APLType:
        if not array.elements:
            raise ValueError("Cannot reduce an empty array")
        resolved_axis = normalize_axis(array.rank, axis)
        output_shape = array.shape[:resolved_axis] + array.shape[resolved_axis + 1:]
        result = []
        for output_index in iter_indices(output_shape):
            slice_values = [
                array.get(output_index[:resolved_axis] + (axis_index,) + output_index[resolved_axis:])
                for axis_index in range(array.shape[resolved_axis])
            ]
            accumulator = slice_values[0].deep_copy()
            for value in slice_values[1:]:
                accumulator = function(accumulator, value)
            result.append(accumulator)
        return result[0] if not output_shape else ArrayType(result, output_shape)

    @staticmethod
    def scan(function: Callable[[APLType, APLType], APLType], array: ArrayType, axis: int | None = None) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        values = {}
        output_shape = array.shape[:resolved_axis] + array.shape[resolved_axis + 1:]
        for output_index in iter_indices(output_shape):
            accumulator = None
            for axis_index in range(array.shape[resolved_axis]):
                full_index = output_index[:resolved_axis] + (axis_index,) + output_index[resolved_axis:]
                value = array.get(full_index)
                accumulator = value.deep_copy() if accumulator is None else function(accumulator, value)
                values[full_index] = accumulator.deep_copy()
        return ArrayType([values[index] for index in iter_indices(array.shape)], array.shape)

    @staticmethod
    def reduce_first_axis(function: Callable[[APLType, APLType], APLType], array: ArrayType) -> APLType:
        return ArrayOperations.reduce(function, array, axis=0)

    @staticmethod
    def scan_first_axis(function: Callable[[APLType, APLType], APLType], array: ArrayType) -> ArrayType:
        return ArrayOperations.scan(function, array, axis=0)

    @staticmethod
    def membership(left: APLType, right: APLType) -> APLType:
        left_value = as_apl_value(left)
        right_value = as_apl_value(right)
        haystack = (
            {element.to_python() for element in right_value.elements}
            if isinstance(right_value, ArrayType)
            else {right_value.to_python()}
        )
        if isinstance(left_value, ArrayType):
            return ArrayType([BooleanType(element.to_python() in haystack) for element in left_value.elements], left_value.shape)
        return BooleanType(left_value.to_python() in haystack)

    @staticmethod
    def indices_where(array: ArrayType) -> ArrayType:
        indices = [IntegerType(index) for index, value in enumerate(array.elements) if value.to_boolean()]
        return ArrayType(indices, (len(indices),))

    @staticmethod
    def enclose(value: APLType) -> ArrayType:
        return ArrayType([as_apl_value(value).deep_copy()], (1,))

    @staticmethod
    def partition(mask: ArrayType, array: ArrayType) -> ArrayType:
        if mask.rank != 1 or array.rank != 1 or len(mask.elements) != len(array.elements):
            raise ValueError("Partition requires 1-D arrays with the same length")
        partitions = []
        current = []
        for marker, element in zip(mask.elements, array.elements):
            if marker.to_boolean() and current:
                partitions.append(ArrayType(current, (len(current),)))
                current = []
            current.append(element.deep_copy())
        if current:
            partitions.append(ArrayType(current, (len(current),)))
        return ArrayType(partitions, (len(partitions),))

    @staticmethod
    def disclose(value: APLType) -> APLType:
        apl_value = as_apl_value(value)
        if isinstance(apl_value, ArrayType) and len(apl_value.elements) == 1:
            return apl_value.elements[0].deep_copy()
        return apl_value.deep_copy()

    @staticmethod
    def pick(index: APLType, array: ArrayType) -> APLType:
        index_value = as_apl_value(index)
        if isinstance(index_value, IntegerType):
            return array.elements[index_value.value].deep_copy()
        if isinstance(index_value, ArrayType):
            coordinates = tuple(int(element.to_numeric()) for element in index_value.elements)
            return array.get(coordinates).deep_copy()
        raise TypeError("Pick requires an integer or index vector")

    @staticmethod
    def grade_up(array: ArrayType, axis: int | None = None) -> ArrayType:
        return ArrayOperations._grade(array, axis=axis, reverse=False)

    @staticmethod
    def grade_down(array: ArrayType, axis: int | None = None) -> ArrayType:
        return ArrayOperations._grade(array, axis=axis, reverse=True)

    @staticmethod
    def sort_by(keys: ArrayType, values: ArrayType, axis: int | None = None) -> ArrayType:
        return ArrayOperations._apply_grade(keys, values, axis=axis, reverse=False)

    @staticmethod
    def reverse_sort_by(keys: ArrayType, values: ArrayType, axis: int | None = None) -> ArrayType:
        return ArrayOperations._apply_grade(keys, values, axis=axis, reverse=True)

    @staticmethod
    def _transform_axis(array: ArrayType, axis: int, coordinate_map):
        result = []
        for index in iter_indices(array.shape):
            source_index = list(index)
            source_index[axis] = coordinate_map(index[axis], array.shape[axis])
            result.append(array.get(tuple(source_index)).deep_copy())
        return ArrayType(result, array.shape)

    @staticmethod
    def _grade(array: ArrayType, axis: int | None = None, reverse: bool = False) -> ArrayType:
        resolved_axis = normalize_axis(array.rank, axis)
        values = {}
        output_shape = array.shape[:resolved_axis] + array.shape[resolved_axis + 1:]
        for output_index in iter_indices(output_shape):
            slice_values = [
                array.get(output_index[:resolved_axis] + (axis_index,) + output_index[resolved_axis:])
                for axis_index in range(array.shape[resolved_axis])
            ]
            order = sorted(
                range(len(slice_values)),
                key=lambda idx: slice_values[idx].to_python(),
                reverse=reverse,
            )
            for axis_index, source in enumerate(order):
                full_index = output_index[:resolved_axis] + (axis_index,) + output_index[resolved_axis:]
                values[full_index] = IntegerType(source)
        return ArrayType([values[index] for index in iter_indices(array.shape)], array.shape)

    @staticmethod
    def _apply_grade(keys: ArrayType, values: ArrayType, axis: int | None = None, reverse: bool = False) -> ArrayType:
        if keys.shape != values.shape:
            raise ValueError("Sort-by requires keys and values with the same shape")
        grade = ArrayOperations._grade(keys, axis=axis, reverse=reverse)
        resolved_axis = normalize_axis(keys.rank, axis)
        sorted_values = {}
        output_shape = keys.shape[:resolved_axis] + keys.shape[resolved_axis + 1:]
        for output_index in iter_indices(output_shape):
            for axis_index in range(keys.shape[resolved_axis]):
                grade_index = output_index[:resolved_axis] + (axis_index,) + output_index[resolved_axis:]
                source_axis = int(grade.get(grade_index).to_numeric())
                source_index = output_index[:resolved_axis] + (source_axis,) + output_index[resolved_axis:]
                sorted_values[grade_index] = values.get(source_index).deep_copy()
        return ArrayType([sorted_values[index] for index in iter_indices(values.shape)], values.shape)
