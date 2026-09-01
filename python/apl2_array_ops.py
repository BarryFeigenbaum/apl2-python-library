"""APL2 Array Operations for Python"""
import math
from typing import List, Tuple
from apl2_types import APLType, ArrayType, IntegerType, FloatingPointType
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
    def reverse(array: ArrayType) -> ArrayType:
        elements = list(reversed(array.elements))
        return ArrayType(elements, array.shape)

    @staticmethod
    def rotate(array: ArrayType, n: int) -> ArrayType:
        elements = array.elements
        size = len(elements)
        if size == 0:
            return array
        n = n % size
        if n < 0:
            n += size
        rotated = elements[-n:] + elements[:-n]
        return ArrayType(rotated, array.shape)

    @staticmethod
    def transpose(array: ArrayType) -> ArrayType:
        if array.rank != 2:
            raise ValueError("Transpose only works on 2-D arrays")
        return array.transpose()

    @staticmethod
    def concatenate(left: ArrayType, right: ArrayType) -> ArrayType:
        combined = left.elements + right.elements
        new_shape = list(left.shape)
        new_shape[0] = left.shape[0] + right.shape[0]
        return ArrayType(combined, tuple(new_shape))

    @staticmethod
    def take(array: ArrayType, n: int) -> ArrayType:
        elements = array.elements
        size = min(n, len(elements))
        taken = elements[:size]
        # Pad with zeros if necessary
        if n > len(elements):
            taken.extend([IntegerType(0)] * (n - len(elements)))
        new_shape = list(array.shape)
        new_shape[0] = n
        return ArrayType(taken, tuple(new_shape))

    @staticmethod
    def drop(array: ArrayType, n: int) -> ArrayType:
        elements = array.elements
        size = max(0, len(elements) - n)
        start = min(n, len(elements))
        dropped = elements[start:]
        new_shape = list(array.shape)
        new_shape[0] = size
        return ArrayType(dropped, tuple(new_shape))

    @staticmethod
    def ravel(array: ArrayType) -> ArrayType:
        return array.flatten()

    @staticmethod
    def count(array: ArrayType) -> int:
        return sum(1 for elem in array.elements if isinstance(elem, APLType) and elem.to_boolean())

    @staticmethod
    def sum(array: ArrayType) -> APLType:
        if not array.elements:
            return IntegerType(0)
        result = array.elements[0].deep_copy()
        for i in range(1, len(array.elements)):
            result = MathOperations.add(result, array.elements[i])
        return result

    @staticmethod
    def product(array: ArrayType) -> APLType:
        if not array.elements:
            return IntegerType(1)
        result = array.elements[0].deep_copy()
        for i in range(1, len(array.elements)):
            result = MathOperations.multiply(result, array.elements[i])
        return result

    @staticmethod
    def maximum(array: ArrayType) -> APLType:
        if not array.elements:
            raise ValueError("Cannot find maximum of empty array")
        result = array.elements[0]
        for i in range(1, len(array.elements)):
            result = MathOperations.max(result, array.elements[i])
        return result

    @staticmethod
    def minimum(array: ArrayType) -> APLType:
        if not array.elements:
            raise ValueError("Cannot find minimum of empty array")
        result = array.elements[0]
        for i in range(1, len(array.elements)):
            result = MathOperations.min(result, array.elements[i])
        return result
