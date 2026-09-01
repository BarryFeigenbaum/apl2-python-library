"""APL2 Type System for Python"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
import math
from decimal import Decimal
from numbers import Number


class APLType(ABC):
    """Base class for all APL2 data types."""

    @abstractmethod
    def get_type_name(self) -> str:
        pass

    @abstractmethod
    def deep_copy(self):
        pass

    @abstractmethod
    def get_rank(self) -> int:
        pass

    @abstractmethod
    def get_shape(self) -> Tuple[int, ...]:
        pass


class Scalar(APLType):
    """Base class for all scalar types."""

    def get_rank(self) -> int:
        return 0

    def get_shape(self) -> Tuple[int, ...]:
        return ()

    @abstractmethod
    def to_numeric(self) -> float:
        pass

    @abstractmethod
    def to_boolean(self) -> bool:
        pass

    @abstractmethod
    def to_character(self) -> str:
        pass


class BooleanType(Scalar):
    """Represents an APL2 Boolean value."""

    def __init__(self, value: bool):
        self.value = bool(value)

    def get_type_name(self) -> str:
        return "Boolean"

    def deep_copy(self):
        return BooleanType(self.value)

    def to_numeric(self) -> float:
        return 1.0 if self.value else 0.0

    def to_boolean(self) -> bool:
        return self.value

    def to_character(self) -> str:
        return '1' if self.value else '0'

    def __eq__(self, other):
        return isinstance(other, BooleanType) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(int(self.value))


class IntegerType(Scalar):
    """Represents an APL2 Integer value."""

    def __init__(self, value: int):
        self.value = int(value)

    def get_type_name(self) -> str:
        return "Integer"

    def deep_copy(self):
        return IntegerType(self.value)

    def to_numeric(self) -> float:
        return float(self.value)

    def to_boolean(self) -> bool:
        return self.value != 0

    def to_character(self) -> str:
        return chr(self.value)

    def __eq__(self, other):
        return isinstance(other, IntegerType) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)


class FloatingPointType(Scalar):
    """Represents an APL2 Floating Point value."""

    EPSILON = 1e-15

    def __init__(self, value: float):
        self.value = float(value)

    def get_type_name(self) -> str:
        return "FloatingPoint"

    def deep_copy(self):
        return FloatingPointType(self.value)

    def to_numeric(self) -> float:
        return self.value

    def to_boolean(self) -> bool:
        return abs(self.value) > self.EPSILON

    def to_character(self) -> str:
        return chr(int(self.value))

    def __eq__(self, other):
        if isinstance(other, FloatingPointType):
            return abs(self.value - other.value) < self.EPSILON
        return False

    def __hash__(self):
        return hash(round(self.value, 15))

    def __repr__(self):
        return str(self.value)


class ComplexType(Scalar):
    """Represents an APL2 Complex number."""

    EPSILON = 1e-15

    def __init__(self, real: float, imaginary: float):
        self.real = float(real)
        self.imaginary = float(imaginary)

    def get_type_name(self) -> str:
        return "Complex"

    def deep_copy(self):
        return ComplexType(self.real, self.imaginary)

    def to_numeric(self) -> float:
        return math.sqrt(self.real**2 + self.imaginary**2)

    def to_boolean(self) -> bool:
        return abs(self.real) > self.EPSILON or abs(self.imaginary) > self.EPSILON

    def to_character(self) -> str:
        return chr(int(self.real))

    def add(self, other: 'ComplexType') -> 'ComplexType':
        return ComplexType(self.real + other.real, self.imaginary + other.imaginary)

    def subtract(self, other: 'ComplexType') -> 'ComplexType':
        return ComplexType(self.real - other.real, self.imaginary - other.imaginary)

    def multiply(self, other: 'ComplexType') -> 'ComplexType':
        real = self.real * other.real - self.imaginary * other.imaginary
        imag = self.real * other.imaginary + self.imaginary * other.real
        return ComplexType(real, imag)

    def divide(self, other: 'ComplexType') -> 'ComplexType':
        denominator = other.real**2 + other.imaginary**2
        real = (self.real * other.real + self.imaginary * other.imaginary) / denominator
        imag = (self.imaginary * other.real - self.real * other.imaginary) / denominator
        return ComplexType(real, imag)

    def conjugate(self) -> 'ComplexType':
        return ComplexType(self.real, -self.imaginary)

    def __eq__(self, other):
        if isinstance(other, ComplexType):
            return abs(self.real - other.real) < self.EPSILON and \
                   abs(self.imaginary - other.imaginary) < self.EPSILON
        return False

    def __repr__(self):
        if abs(self.imaginary) < self.EPSILON:
            return str(self.real)
        return f"{self.real}{'+' if self.imaginary >= 0 else ''}{self.imaginary}i"


class CharacterType(Scalar):
    """Represents an APL2 Character."""

    def __init__(self, value: str):
        if isinstance(value, str) and len(value) == 1:
            self.value = value
        else:
            self.value = str(value)[0] if value else '\0'

    def get_type_name(self) -> str:
        return "Character"

    def deep_copy(self):
        return CharacterType(self.value)

    def to_numeric(self) -> float:
        return float(ord(self.value))

    def to_boolean(self) -> bool:
        return ord(self.value) != 0

    def to_character(self) -> str:
        return self.value

    def __eq__(self, other):
        return isinstance(other, CharacterType) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return repr(self.value)


class StringType(Scalar):
    """Represents an APL2 String."""

    def __init__(self, value: str):
        self.value = str(value) if value is not None else ""

    def get_type_name(self) -> str:
        return "String"

    def deep_copy(self):
        return StringType(self.value)

    def to_numeric(self) -> float:
        try:
            return float(self.value)
        except ValueError:
            return 0.0

    def to_boolean(self) -> bool:
        return len(self.value) > 0

    def to_character(self) -> str:
        return self.value[0] if self.value else '\0'

    def __eq__(self, other):
        return isinstance(other, StringType) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return repr(self.value)


class BigIntegerType(Scalar):
    """Represents an APL2 BigInteger for arbitrary-precision integers."""

    def __init__(self, value: Union[int, str]):
        self.value = int(value)

    def get_type_name(self) -> str:
        return "BigInteger"

    def deep_copy(self):
        return BigIntegerType(self.value)

    def to_numeric(self) -> float:
        return float(self.value)

    def to_boolean(self) -> bool:
        return self.value != 0

    def to_character(self) -> str:
        return chr(self.value % 1114112)  # Unicode limit

    def add(self, other: 'BigIntegerType') -> 'BigIntegerType':
        return BigIntegerType(self.value + other.value)

    def subtract(self, other: 'BigIntegerType') -> 'BigIntegerType':
        return BigIntegerType(self.value - other.value)

    def multiply(self, other: 'BigIntegerType') -> 'BigIntegerType':
        return BigIntegerType(self.value * other.value)

    def divide(self, other: 'BigIntegerType') -> 'BigIntegerType':
        if other.value == 0:
            raise ArithmeticError("Division by zero")
        return BigIntegerType(self.value // other.value)

    def mod(self, other: 'BigIntegerType') -> 'BigIntegerType':
        if other.value == 0:
            raise ArithmeticError("Modulo by zero")
        return BigIntegerType(self.value % other.value)

    def pow(self, exponent: int) -> 'BigIntegerType':
        return BigIntegerType(self.value ** exponent)

    def abs(self) -> 'BigIntegerType':
        return BigIntegerType(abs(self.value))

    def negate(self) -> 'BigIntegerType':
        return BigIntegerType(-self.value)

    def __eq__(self, other):
        return isinstance(other, BigIntegerType) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)


class BigDecimalType(Scalar):
    """Represents an APL2 BigDecimal for arbitrary-precision decimals."""

    def __init__(self, value: Union[float, str, Decimal]):
        if isinstance(value, Decimal):
            self.value = value
        else:
            self.value = Decimal(str(value))

    def get_type_name(self) -> str:
        return "BigDecimal"

    def deep_copy(self):
        return BigDecimalType(self.value)

    def to_numeric(self) -> float:
        return float(self.value)

    def to_boolean(self) -> bool:
        return self.value != 0

    def to_character(self) -> str:
        return chr(int(self.value))

    def add(self, other: 'BigDecimalType') -> 'BigDecimalType':
        return BigDecimalType(self.value + other.value)

    def subtract(self, other: 'BigDecimalType') -> 'BigDecimalType':
        return BigDecimalType(self.value - other.value)

    def multiply(self, other: 'BigDecimalType') -> 'BigDecimalType':
        return BigDecimalType(self.value * other.value)

    def divide(self, other: 'BigDecimalType') -> 'BigDecimalType':
        if other.value == 0:
            raise ArithmeticError("Division by zero")
        return BigDecimalType(self.value / other.value)

    def abs(self) -> 'BigDecimalType':
        return BigDecimalType(abs(self.value))

    def negate(self) -> 'BigDecimalType':
        return BigDecimalType(-self.value)

    def __eq__(self, other):
        return isinstance(other, BigDecimalType) and self.value == other.value

    def __hash__(self):
        return hash(str(self.value))

    def __repr__(self):
        return str(self.value)


class ArrayType(APLType):
    """Represents a multi-dimensional array in APL2."""

    MAX_DEPTH = 15

    def __init__(self, elements: List[APLType], shape: Tuple[int, ...] = None):
        self.elements = list(elements)
        if shape is None:
            self.shape = (len(elements),)
        else:
            self.shape = tuple(shape)
            # Validate shape
            expected_size = 1
            for dim in shape:
                if dim < 0:
                    raise ValueError("Shape dimensions must be non-negative")
                expected_size *= dim
            if len(elements) != expected_size:
                raise ValueError(f"Elements size {len(elements)} doesn't match shape {shape}")
        
        self.rank = len(self.shape)
        self._depth = self._calculate_depth()

    def _calculate_depth(self) -> int:
        if not self.elements:
            return 1
        if not isinstance(self.elements[0], ArrayType):
            return 1
        max_child_depth = max(e._depth for e in self.elements if isinstance(e, ArrayType))
        return max_child_depth + 1

    def get_type_name(self) -> str:
        return "Array"

    def deep_copy(self):
        copied_elements = [e.deep_copy() for e in self.elements]
        return ArrayType(copied_elements, self.shape)

    def get_rank(self) -> int:
        return self.rank

    def get_shape(self) -> Tuple[int, ...]:
        return self.shape

    def get_element(self, *indices: int) -> APLType:
        if len(indices) == 1:
            return self.elements[indices[0]]
        flat_index = self._to_flat_index(*indices)
        return self.elements[flat_index]

    def _to_flat_index(self, *indices: int) -> int:
        flat_index = 0
        multiplier = 1
        for i in range(self.rank - 1, -1, -1):
            if indices[i] < 0 or indices[i] >= self.shape[i]:
                raise IndexError(f"Index {indices} out of bounds for shape {self.shape}")
            flat_index += indices[i] * multiplier
            multiplier *= self.shape[i]
        return flat_index

    def reshape(self, *new_shape: int) -> 'ArrayType':
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        if new_size != len(self.elements):
            raise ValueError(f"Cannot reshape array of size {len(self.elements)} to shape {new_shape}")
        return ArrayType(self.elements, new_shape)

    def flatten(self) -> 'ArrayType':
        if self.rank == 1:
            return ArrayType(self.elements)
        return ArrayType(self.elements)

    def transpose(self) -> 'ArrayType':
        if self.rank != 2:
            raise ValueError("Transpose only works on 2-D arrays")
        rows, cols = self.shape
        transposed = []
        for j in range(cols):
            for i in range(rows):
                transposed.append(self.elements[i * cols + j])
        return ArrayType(transposed, (cols, rows))

    def __repr__(self):
        return f"[{', '.join(str(e) for e in self.elements)}]"

    def __eq__(self, other):
        return isinstance(other, ArrayType) and \
               self.elements == other.elements and \
               self.shape == other.shape
