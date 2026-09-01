"""APL2 Tests for Python Implementation"""
import unittest
import math
from apl2_types import (
    BooleanType, IntegerType, FloatingPointType, ComplexType,
    CharacterType, StringType, BigIntegerType, BigDecimalType, ArrayType
)
from apl2_operations import MathOperations
from apl2_array_ops import ArrayOperations


class TestScalarTypes(unittest.TestCase):
    def test_boolean_type(self):
        true_val = BooleanType(True)
        false_val = BooleanType(False)
        self.assertTrue(true_val.value)
        self.assertFalse(false_val.value)
        self.assertEqual(true_val.to_numeric(), 1.0)
        self.assertEqual(false_val.to_numeric(), 0.0)

    def test_integer_type(self):
        num = IntegerType(42)
        self.assertEqual(num.value, 42)
        self.assertEqual(num.to_numeric(), 42.0)
        self.assertTrue(num.to_boolean())
        self.assertFalse(IntegerType(0).to_boolean())

    def test_floating_point_type(self):
        num = FloatingPointType(3.14)
        self.assertAlmostEqual(num.value, 3.14)
        self.assertTrue(num.to_boolean())
        self.assertFalse(FloatingPointType(0.0).to_boolean())

    def test_complex_type(self):
        c = ComplexType(3.0, 4.0)
        self.assertEqual(c.real, 3.0)
        self.assertEqual(c.imaginary, 4.0)
        self.assertEqual(c.to_numeric(), 5.0)  # magnitude

    def test_string_type(self):
        s = StringType("hello")
        self.assertEqual(s.value, "hello")
        self.assertTrue(s.to_boolean())
        self.assertFalse(StringType("").to_boolean())

    def test_big_integer_type(self):
        big = BigIntegerType("999999999999999999999999999")
        self.assertEqual(big.value, 999999999999999999999999999)
        self.assertTrue(big.to_boolean())


class TestMathOperations(unittest.TestCase):
    def test_add(self):
        result = MathOperations.add(IntegerType(5), IntegerType(3))
        self.assertEqual(result.value, 8)

    def test_subtract(self):
        result = MathOperations.subtract(IntegerType(10), IntegerType(3))
        self.assertEqual(result.value, 7)

    def test_multiply(self):
        result = MathOperations.multiply(IntegerType(6), IntegerType(7))
        self.assertEqual(result.value, 42)

    def test_divide(self):
        result = MathOperations.divide(IntegerType(20), IntegerType(4))
        self.assertAlmostEqual(result.to_numeric(), 5.0)

    def test_divide_by_zero(self):
        with self.assertRaises(ArithmeticError):
            MathOperations.divide(IntegerType(5), IntegerType(0))

    def test_power(self):
        result = MathOperations.power(IntegerType(2), IntegerType(8))
        self.assertEqual(result.to_numeric(), 256.0)

    def test_negate(self):
        result = MathOperations.negate(IntegerType(42))
        self.assertEqual(result.value, -42)

    def test_abs(self):
        result = MathOperations.abs(IntegerType(-42))
        self.assertEqual(result.value, 42)

    def test_sqrt(self):
        result = MathOperations.sqrt(FloatingPointType(16.0))
        self.assertEqual(result.to_numeric(), 4.0)

    def test_sin(self):
        result = MathOperations.sin(FloatingPointType(0.0))
        self.assertAlmostEqual(result.to_numeric(), 0.0, places=4)

    def test_max(self):
        result = MathOperations.max(IntegerType(5), IntegerType(10))
        self.assertEqual(result.to_numeric(), 10.0)

    def test_min(self):
        result = MathOperations.min(IntegerType(5), IntegerType(10))
        self.assertEqual(result.to_numeric(), 5.0)


class TestArrayOperations(unittest.TestCase):
    def test_create_array(self):
        elements = [IntegerType(1), IntegerType(2), IntegerType(3)]
        array = ArrayType(elements)
        self.assertEqual(array.rank, 1)
        self.assertEqual(array.shape, (3,))
        self.assertEqual(array.size(), 3)

    def test_sum(self):
        elements = [IntegerType(1), IntegerType(2), IntegerType(3)]
        array = ArrayType(elements)
        result = ArrayOperations.sum(array)
        self.assertEqual(result.value, 6)

    def test_product(self):
        elements = [IntegerType(2), IntegerType(3), IntegerType(4)]
        array = ArrayType(elements)
        result = ArrayOperations.product(array)
        self.assertEqual(result.value, 24)

    def test_maximum(self):
        elements = [IntegerType(3), IntegerType(1), IntegerType(5)]
        array = ArrayType(elements)
        result = ArrayOperations.maximum(array)
        self.assertEqual(result.to_numeric(), 5.0)

    def test_minimum(self):
        elements = [IntegerType(3), IntegerType(1), IntegerType(5)]
        array = ArrayType(elements)
        result = ArrayOperations.minimum(array)
        self.assertEqual(result.to_numeric(), 1.0)

    def test_reverse(self):
        elements = [IntegerType(1), IntegerType(2), IntegerType(3)]
        array = ArrayType(elements)
        reversed_array = ArrayOperations.reverse(array)
        self.assertEqual(reversed_array.elements[0].value, 3)
        self.assertEqual(reversed_array.elements[2].value, 1)

    def test_rotate(self):
        elements = [IntegerType(1), IntegerType(2), IntegerType(3)]
        array = ArrayType(elements)
        rotated = ArrayOperations.rotate(array, 1)
        self.assertEqual(rotated.elements[0].value, 3)
        self.assertEqual(rotated.elements[1].value, 1)
        self.assertEqual(rotated.elements[2].value, 2)


if __name__ == '__main__':
    unittest.main()
