"""APL2 scalar and element-wise operations for Python."""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from decimal import Decimal
from typing import Callable

from apl2_types import (
    APLType,
    ArrayType,
    BigDecimalType,
    BooleanType,
    ComplexType,
    FloatingPointType,
    IntegerType,
    StringType,
    as_apl_value,
)


class MathOperations:
    """Scalar and element-wise APL2 operations."""

    @staticmethod
    def _numeric_value(value: APLType):
        if isinstance(value, BigDecimalType):
            return value.value
        if isinstance(value, ComplexType):
            return value.value
        if isinstance(value, (BooleanType, IntegerType)):
            return value.value
        if isinstance(value, FloatingPointType):
            return value.value
        raise TypeError(f"{value.__class__.__name__} is not numeric")

    @staticmethod
    def _wrap_number(value):
        if isinstance(value, APLType):
            return value
        if isinstance(value, Decimal):
            if value == value.to_integral_value():
                return IntegerType(int(value))
            return BigDecimalType(value)
        if isinstance(value, complex):
            return ComplexType(value.real, value.imag)
        if isinstance(value, bool):
            return BooleanType(value)
        if isinstance(value, int):
            return IntegerType(value)
        if isinstance(value, float):
            if value.is_integer():
                return IntegerType(int(value))
            return FloatingPointType(value)
        raise TypeError(f"Unsupported numeric result: {value!r}")

    @staticmethod
    def _binary(left, right, operation: Callable[[APLType, APLType], APLType]):
        left_value = as_apl_value(left)
        right_value = as_apl_value(right)
        if isinstance(left_value, ArrayType) and isinstance(right_value, ArrayType):
            if left_value.shape != right_value.shape:
                raise ValueError("Array arguments must have the same shape")
            return ArrayType(
                [MathOperations._binary(l, r, operation) for l, r in zip(left_value.elements, right_value.elements)],
                left_value.shape,
            )
        if isinstance(left_value, ArrayType):
            return ArrayType(
                [MathOperations._binary(element, right_value, operation) for element in left_value.elements],
                left_value.shape,
            )
        if isinstance(right_value, ArrayType):
            return ArrayType(
                [MathOperations._binary(left_value, element, operation) for element in right_value.elements],
                right_value.shape,
            )
        return operation(left_value, right_value)

    @staticmethod
    def _unary(value, operation: Callable[[APLType], APLType]):
        apl_value = as_apl_value(value)
        if isinstance(apl_value, ArrayType):
            return ArrayType([MathOperations._unary(element, operation) for element in apl_value.elements], apl_value.shape)
        return operation(apl_value)

    @staticmethod
    def add(left, right):
        return MathOperations._binary(
            left,
            right,
            lambda l, r: MathOperations._wrap_number(MathOperations._numeric_value(l) + MathOperations._numeric_value(r)),
        )

    @staticmethod
    def subtract(left, right):
        return MathOperations._binary(
            left,
            right,
            lambda l, r: MathOperations._wrap_number(MathOperations._numeric_value(l) - MathOperations._numeric_value(r)),
        )

    @staticmethod
    def multiply(left, right):
        return MathOperations._binary(
            left,
            right,
            lambda l, r: MathOperations._wrap_number(MathOperations._numeric_value(l) * MathOperations._numeric_value(r)),
        )

    @staticmethod
    def divide(left, right):
        def operation(l, r):
            divisor = MathOperations._numeric_value(r)
            if divisor == 0:
                raise ArithmeticError("Division by zero")
            return MathOperations._wrap_number(MathOperations._numeric_value(l) / divisor)

        return MathOperations._binary(left, right, operation)

    @staticmethod
    def power(left, right):
        return MathOperations._binary(
            left,
            right,
            lambda l, r: MathOperations._wrap_number(MathOperations._numeric_value(l) ** MathOperations._numeric_value(r)),
        )

    @staticmethod
    def conjugate(value):
        return MathOperations._unary(
            value,
            lambda item: MathOperations._wrap_number(
                MathOperations._numeric_value(item).conjugate()
                if isinstance(item, ComplexType)
                else MathOperations._numeric_value(item)
            ),
        )

    @staticmethod
    def negate(value):
        return MathOperations._unary(value, lambda item: MathOperations._wrap_number(-MathOperations._numeric_value(item)))

    @staticmethod
    def signum(value):
        def operation(item):
            numeric = MathOperations._numeric_value(item)
            if numeric == 0:
                return IntegerType(0)
            if isinstance(numeric, complex):
                return MathOperations._wrap_number(numeric / abs(numeric))
            return IntegerType(1 if numeric > 0 else -1)

        return MathOperations._unary(value, operation)

    @staticmethod
    def reciprocal(value):
        def operation(item):
            numeric = MathOperations._numeric_value(item)
            if numeric == 0:
                raise ArithmeticError("Division by zero")
            return MathOperations._wrap_number(1 / numeric)

        return MathOperations._unary(value, operation)

    @staticmethod
    def residue(left, right):
        return MathOperations._binary(
            left,
            right,
            lambda l, r: MathOperations._wrap_number(MathOperations._numeric_value(r) % MathOperations._numeric_value(l)),
        )

    @staticmethod
    def magnitude(value):
        return MathOperations._unary(value, lambda item: MathOperations._wrap_number(abs(MathOperations._numeric_value(item))))

    abs = magnitude

    @staticmethod
    def exponential(value):
        return MathOperations._unary(value, lambda item: MathOperations._wrap_number(math.exp(MathOperations._numeric_value(item))))

    @staticmethod
    def natural_log(value):
        return MathOperations._unary(value, lambda item: MathOperations._wrap_number(math.log(MathOperations._numeric_value(item))))

    @staticmethod
    def logarithm(base, value):
        return MathOperations._binary(
            base,
            value,
            lambda l, r: MathOperations._wrap_number(
                math.log(MathOperations._numeric_value(r), MathOperations._numeric_value(l))
            ),
        )

    @staticmethod
    def factorial(value):
        def operation(item):
            numeric = MathOperations._numeric_value(item)
            if isinstance(numeric, complex):
                raise TypeError("Factorial does not support complex values")
            if float(numeric).is_integer() and numeric >= 0:
                return IntegerType(math.factorial(int(numeric)))
            return FloatingPointType(math.gamma(float(numeric) + 1.0))

        return MathOperations._unary(value, operation)

    @staticmethod
    def binomial(left, right):
        def operation(l, r):
            left_value = float(MathOperations._numeric_value(l))
            right_value = float(MathOperations._numeric_value(r))
            result = math.gamma(right_value + 1.0) / (
                math.gamma(left_value + 1.0) * math.gamma(right_value - left_value + 1.0)
            )
            return MathOperations._wrap_number(result)

        return MathOperations._binary(left, right, operation)

    @staticmethod
    def circle_functions(value, selector: int = 1):
        return MathOperations.circle(selector, value)

    @staticmethod
    def circle(selector, value):
        def operation(item):
            numeric = float(MathOperations._numeric_value(item))
            mapping = {
                -7: math.atanh,
                -6: math.acosh,
                -5: math.asinh,
                -3: math.atan,
                -2: math.acos,
                -1: math.asin,
                1: math.sin,
                2: math.cos,
                3: math.tan,
                5: math.sinh,
                6: math.cosh,
                7: math.tanh,
            }
            if selector not in mapping:
                raise ValueError(f"Unsupported circle selector: {selector}")
            return FloatingPointType(mapping[selector](numeric))

        return MathOperations._unary(value, operation)

    @staticmethod
    def sqrt(value):
        return MathOperations._unary(value, lambda item: MathOperations._wrap_number(math.sqrt(MathOperations._numeric_value(item))))

    @staticmethod
    def sin(value):
        return MathOperations.circle(1, value)

    @staticmethod
    def equal(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(l.to_python() == r.to_python()))

    @staticmethod
    def not_equal(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(l.to_python() != r.to_python()))

    @staticmethod
    def less_than(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(MathOperations._numeric_value(l) < MathOperations._numeric_value(r)))

    @staticmethod
    def less_equal(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(MathOperations._numeric_value(l) <= MathOperations._numeric_value(r)))

    @staticmethod
    def greater_than(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(MathOperations._numeric_value(l) > MathOperations._numeric_value(r)))

    @staticmethod
    def greater_equal(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(MathOperations._numeric_value(l) >= MathOperations._numeric_value(r)))

    @staticmethod
    def logical_and(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(l.to_boolean() and r.to_boolean()))

    @staticmethod
    def logical_or(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(l.to_boolean() or r.to_boolean()))

    @staticmethod
    def logical_nand(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(not (l.to_boolean() and r.to_boolean())))

    @staticmethod
    def logical_nor(left, right):
        return MathOperations._binary(left, right, lambda l, r: BooleanType(not (l.to_boolean() or r.to_boolean())))

    @staticmethod
    def logical_not(value):
        return MathOperations._unary(value, lambda item: BooleanType(not item.to_boolean()))

    @staticmethod
    def without(left, right):
        left_value = as_apl_value(left)
        right_value = as_apl_value(right)
        if isinstance(left_value, ArrayType):
            exclusions = (
                {element.to_python() for element in right_value.elements}
                if isinstance(right_value, ArrayType)
                else {right_value.to_python()}
            )
            return ArrayType(
                [element.deep_copy() for element in left_value.elements if element.to_python() not in exclusions],
                (sum(1 for element in left_value.elements if element.to_python() not in exclusions),),
            )
        if isinstance(right_value, ArrayType):
            return BooleanType(left_value.to_python() not in {element.to_python() for element in right_value.elements})
        return BooleanType(left_value.to_python() != right_value.to_python())

    @staticmethod
    def roll(value):
        limit = int(MathOperations._numeric_value(as_apl_value(value)))
        if limit <= 0:
            raise ValueError("Roll requires a positive integer")
        return IntegerType(random.randint(1, limit))

    @staticmethod
    def deal(count, limit):
        count_value = int(MathOperations._numeric_value(as_apl_value(count)))
        limit_value = int(MathOperations._numeric_value(as_apl_value(limit)))
        if count_value < 0 or limit_value <= 0 or count_value > limit_value:
            raise ValueError("Deal requires 0 <= count <= limit and limit > 0")
        return ArrayType([IntegerType(value) for value in random.sample(range(1, limit_value + 1), count_value)], (count_value,))

    @staticmethod
    def format(value):
        return StringType(str(as_apl_value(value).to_python()))

    @staticmethod
    def format_with_pattern(pattern, value):
        pattern_value = as_apl_value(pattern)
        if not isinstance(pattern_value, StringType):
            raise TypeError("Format pattern must be a string")
        return StringType(format(as_apl_value(value).to_python(), pattern_value.value))

    @staticmethod
    def max(left, right):
        return MathOperations._binary(left, right, lambda l, r: l.deep_copy() if MathOperations._numeric_value(l) >= MathOperations._numeric_value(r) else r.deep_copy())

    @staticmethod
    def min(left, right):
        return MathOperations._binary(left, right, lambda l, r: l.deep_copy() if MathOperations._numeric_value(l) <= MathOperations._numeric_value(r) else r.deep_copy())

    @staticmethod
    def iota(value):
        count = int(MathOperations._numeric_value(as_apl_value(value)))
        if count < 0:
            raise ValueError("Iota requires a non-negative integer")
        return ArrayType([IntegerType(index) for index in range(count)], (count,))

    @staticmethod
    def index_of(left, right):
        left_array = as_apl_value(left)
        right_value = as_apl_value(right)
        if not isinstance(left_array, ArrayType):
            raise TypeError("Index-of requires the left argument to be an array")
        lookup = [element.to_python() for element in left_array.elements]

        def find(value):
            python_value = value.to_python()
            try:
                return IntegerType(lookup.index(python_value))
            except ValueError:
                return IntegerType(len(lookup))

        if isinstance(right_value, ArrayType):
            return ArrayType([find(element) for element in right_value.elements], right_value.shape)
        return find(right_value)

    @staticmethod
    def where(left, right):
        left_array = as_apl_value(left)
        right_value = as_apl_value(right)
        if not isinstance(left_array, ArrayType):
            raise TypeError("Where requires the left argument to be an array")
        sorted_values = [float(MathOperations._numeric_value(element)) for element in left_array.elements]

        def locate(value):
            return IntegerType(bisect_left(sorted_values, float(MathOperations._numeric_value(value))))

        if isinstance(right_value, ArrayType):
            return ArrayType([locate(element) for element in right_value.elements], right_value.shape)
        return locate(right_value)
