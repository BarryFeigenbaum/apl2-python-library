"""APL2 Math Operations for Python"""
import math
from typing import Union
from apl2_types import (
    APLType, Scalar, IntegerType, FloatingPointType, ComplexType,
    BigIntegerType, BigDecimalType
)


class MathOperations:
    """Core APL2 mathematical operations."""

    @staticmethod
    def add(left: APLType, right: APLType) -> APLType:
        if isinstance(left, IntegerType) and isinstance(right, IntegerType):
            return IntegerType(left.value + right.value)
        if isinstance(left, ComplexType) and isinstance(right, ComplexType):
            return left.add(right)
        if isinstance(left, BigIntegerType) and isinstance(right, BigIntegerType):
            return left.add(right)
        if isinstance(left, BigDecimalType) and isinstance(right, BigDecimalType):
            return left.add(right)
        return FloatingPointType(MathOperations._to_numeric(left) + MathOperations._to_numeric(right))

    @staticmethod
    def subtract(left: APLType, right: APLType) -> APLType:
        if isinstance(left, IntegerType) and isinstance(right, IntegerType):
            return IntegerType(left.value - right.value)
        if isinstance(left, ComplexType) and isinstance(right, ComplexType):
            return left.subtract(right)
        if isinstance(left, BigIntegerType) and isinstance(right, BigIntegerType):
            return left.subtract(right)
        return FloatingPointType(MathOperations._to_numeric(left) - MathOperations._to_numeric(right))

    @staticmethod
    def multiply(left: APLType, right: APLType) -> APLType:
        if isinstance(left, IntegerType) and isinstance(right, IntegerType):
            return IntegerType(left.value * right.value)
        if isinstance(left, ComplexType) and isinstance(right, ComplexType):
            return left.multiply(right)
        if isinstance(left, BigIntegerType) and isinstance(right, BigIntegerType):
            return left.multiply(right)
        return FloatingPointType(MathOperations._to_numeric(left) * MathOperations._to_numeric(right))

    @staticmethod
    def divide(left: APLType, right: APLType) -> APLType:
        rval = MathOperations._to_numeric(right)
        if abs(rval) < 1e-15:
            raise ArithmeticError("Division by zero")
        if isinstance(left, IntegerType) and isinstance(right, IntegerType):
            return FloatingPointType(left.value / right.value)
        if isinstance(left, ComplexType) and isinstance(right, ComplexType):
            return left.divide(right)
        return FloatingPointType(MathOperations._to_numeric(left) / rval)

    @staticmethod
    def power(left: APLType, right: APLType) -> APLType:
        base = MathOperations._to_numeric(left)
        exponent = MathOperations._to_numeric(right)
        return FloatingPointType(base ** exponent)

    @staticmethod
    def negate(operand: APLType) -> APLType:
        if isinstance(operand, IntegerType):
            return IntegerType(-operand.value)
        if isinstance(operand, ComplexType):
            return operand.multiply(ComplexType(-1, 0))
        if isinstance(operand, BigIntegerType):
            return operand.negate()
        return FloatingPointType(-MathOperations._to_numeric(operand))

    @staticmethod
    def abs(operand: APLType) -> APLType:
        if isinstance(operand, IntegerType):
            return IntegerType(abs(operand.value))
        if isinstance(operand, ComplexType):
            return FloatingPointType(operand.to_numeric())
        if isinstance(operand, BigIntegerType):
            return operand.abs()
        return FloatingPointType(abs(MathOperations._to_numeric(operand)))

    @staticmethod
    def sqrt(operand: APLType) -> APLType:
        value = MathOperations._to_numeric(operand)
        if value < 0:
            return ComplexType(0, math.sqrt(-value))
        return FloatingPointType(math.sqrt(value))

    @staticmethod
    def ceiling(operand: APLType) -> APLType:
        if isinstance(operand, IntegerType):
            return operand
        return IntegerType(math.ceil(MathOperations._to_numeric(operand)))

    @staticmethod
    def floor(operand: APLType) -> APLType:
        if isinstance(operand, IntegerType):
            return operand
        return IntegerType(math.floor(MathOperations._to_numeric(operand)))

    @staticmethod
    def sign(operand: APLType) -> APLType:
        value = MathOperations._to_numeric(operand)
        if value > 0:
            return IntegerType(1)
        elif value < 0:
            return IntegerType(-1)
        else:
            return IntegerType(0)

    @staticmethod
    def log(operand: APLType) -> APLType:
        value = MathOperations._to_numeric(operand)
        if value <= 0:
            raise ArithmeticError("Logarithm of non-positive number")
        return FloatingPointType(math.log(value))

    @staticmethod
    def exp(operand: APLType) -> APLType:
        return FloatingPointType(math.exp(MathOperations._to_numeric(operand)))

    @staticmethod
    def sin(operand: APLType) -> APLType:
        return FloatingPointType(math.sin(MathOperations._to_numeric(operand)))

    @staticmethod
    def cos(operand: APLType) -> APLType:
        return FloatingPointType(math.cos(MathOperations._to_numeric(operand)))

    @staticmethod
    def tan(operand: APLType) -> APLType:
        return FloatingPointType(math.tan(MathOperations._to_numeric(operand)))

    @staticmethod
    def max(left: APLType, right: APLType) -> APLType:
        lval = MathOperations._to_numeric(left)
        rval = MathOperations._to_numeric(right)
        return FloatingPointType(max(lval, rval))

    @staticmethod
    def min(left: APLType, right: APLType) -> APLType:
        lval = MathOperations._to_numeric(left)
        rval = MathOperations._to_numeric(right)
        return FloatingPointType(min(lval, rval))

    @staticmethod
    def _to_numeric(value: APLType) -> float:
        if isinstance(value, Scalar):
            return value.to_numeric()
        raise TypeError("Cannot convert non-scalar to numeric")
