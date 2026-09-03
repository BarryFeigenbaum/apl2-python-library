"""APL2 tests for the Python implementation."""

import math
import random
import unittest

from apl2_array_ops import ArrayOperations
from apl2_operations import MathOperations
from apl2_types import (
    ArrayType,
    BigDecimalType,
    BigIntegerType,
    BooleanType,
    CharacterType,
    ComplexType,
    FloatingPointType,
    IntegerType,
    StringType,
)


def values_of(array):
    return array.to_python() if isinstance(array, ArrayType) else array.to_python()


class TestScalarTypes(unittest.TestCase):
    def test_scalar_types(self):
        self.assertTrue(BooleanType(True).to_boolean())
        self.assertEqual(IntegerType(42).to_numeric(), 42.0)
        self.assertAlmostEqual(FloatingPointType(3.14).to_numeric(), 3.14)
        self.assertEqual(ComplexType(3.0, 4.0).to_numeric(), 5.0)
        self.assertEqual(CharacterType("a").to_python(), "a")
        self.assertEqual(StringType("hello").to_python(), "hello")
        self.assertEqual(BigIntegerType("999999999999999999").value, 999999999999999999)
        self.assertAlmostEqual(BigDecimalType("12.5").to_numeric(), 12.5)

    def test_array_type_shape_and_reshape(self):
        array = ArrayType([1, 2, 3], (3,))
        self.assertEqual(array.shape, (3,))
        self.assertEqual(array.reshape(2, 2).to_python(), [[1, 2], [3, 1]])
        self.assertEqual(ArrayType([], (0,)).reshape(2, 2).to_python(), [[0, 0], [0, 0]])


class TestMathOperations(unittest.TestCase):
    def test_arithmetic_and_unary_primitives(self):
        self.assertEqual(MathOperations.add(5, 3).to_python(), 8)
        self.assertEqual(MathOperations.subtract(10, 3).to_python(), 7)
        self.assertEqual(MathOperations.multiply(6, 7).to_python(), 42)
        self.assertEqual(MathOperations.divide(20, 4).to_python(), 5)
        self.assertEqual(MathOperations.power(2, 8).to_python(), 256)
        self.assertEqual(MathOperations.conjugate(ComplexType(3, 4)).to_python(), complex(3, -4))
        self.assertEqual(MathOperations.negate(42).to_python(), -42)
        self.assertEqual(MathOperations.signum(-9).to_python(), -1)
        self.assertAlmostEqual(MathOperations.reciprocal(4).to_numeric(), 0.25)
        self.assertEqual(MathOperations.residue(3, 10).to_python(), 1)
        self.assertEqual(MathOperations.magnitude(-42).to_python(), 42)

    def test_exponential_logarithmic_and_combinatoric_primitives(self):
        self.assertAlmostEqual(MathOperations.exponential(1).to_numeric(), math.e, places=5)
        self.assertAlmostEqual(MathOperations.natural_log(math.e).to_numeric(), 1.0, places=5)
        self.assertAlmostEqual(MathOperations.logarithm(2, 8).to_numeric(), 3.0, places=5)
        self.assertEqual(MathOperations.factorial(5).to_python(), 120)
        self.assertEqual(MathOperations.binomial(2, 5).to_python(), 10)
        self.assertAlmostEqual(MathOperations.circle_functions(0.0).to_numeric(), 0.0, places=5)
        self.assertAlmostEqual(MathOperations.circle(2, 0.0).to_numeric(), 1.0, places=5)

    def test_comparison_and_logical_primitives(self):
        self.assertTrue(MathOperations.equal(4, 4).to_boolean())
        self.assertTrue(MathOperations.not_equal(4, 5).to_boolean())
        self.assertTrue(MathOperations.less_than(4, 5).to_boolean())
        self.assertTrue(MathOperations.less_equal(5, 5).to_boolean())
        self.assertTrue(MathOperations.greater_than(5, 4).to_boolean())
        self.assertTrue(MathOperations.greater_equal(5, 5).to_boolean())
        self.assertFalse(MathOperations.logical_and(True, False).to_boolean())
        self.assertTrue(MathOperations.logical_or(True, False).to_boolean())
        self.assertFalse(MathOperations.logical_nand(True, True).to_boolean())
        self.assertTrue(MathOperations.logical_nor(False, False).to_boolean())
        self.assertFalse(MathOperations.logical_not(True).to_boolean())
        self.assertEqual(MathOperations.without(ArrayType([1, 2, 3, 2]), ArrayType([2])).to_python(), [1, 3])

    def test_index_random_and_format_primitives(self):
        random.seed(0)
        self.assertEqual(MathOperations.iota(5).to_python(), [0, 1, 2, 3, 4])
        self.assertEqual(MathOperations.index_of(ArrayType([10, 20, 30]), ArrayType([20, 99])).to_python(), [1, 3])
        self.assertEqual(MathOperations.where(ArrayType([10, 20, 30]), ArrayType([5, 20, 35])).to_python(), [0, 1, 3])
        self.assertEqual(MathOperations.roll(6).to_python(), 4)
        self.assertEqual(MathOperations.deal(3, 5).to_python(), [4, 1, 2])
        self.assertEqual(MathOperations.format(123).to_python(), "123")
        self.assertEqual(MathOperations.format_with_pattern(StringType(".2f"), 3.14159).to_python(), "3.14")

    def test_array_broadcasting(self):
        result = MathOperations.add(ArrayType([1, 2, 3]), 10)
        self.assertEqual(result.to_python(), [11, 12, 13])


class TestArrayOperations(unittest.TestCase):
    def setUp(self):
        self.vector = ArrayType([1, 2, 3, 4], (4,))
        self.matrix = ArrayType([1, 2, 3, 4, 5, 6], (2, 3))

    def test_shape_ravel_reverse_and_rotate_with_axis_support(self):
        self.assertEqual(ArrayOperations.shape(self.matrix).to_python(), [2, 3])
        self.assertEqual(ArrayOperations.ravel(self.matrix).to_python(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(ArrayOperations.reverse(self.matrix).to_python(), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(ArrayOperations.reverse(self.matrix, axis=0).to_python(), [[4, 5, 6], [1, 2, 3]])
        self.assertEqual(ArrayOperations.reverse(self.matrix, axis=-1).to_python(), [[3, 2, 1], [6, 5, 4]])
        self.assertEqual(ArrayOperations.rotate(self.matrix, 1).to_python(), [[3, 1, 2], [6, 4, 5]])
        self.assertEqual(ArrayOperations.rotate_first_axis(self.matrix, 1).to_python(), [[4, 5, 6], [1, 2, 3]])

    def test_transpose_take_drop_first_split_and_catenate(self):
        self.assertEqual(ArrayOperations.transpose(self.matrix).to_python(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(ArrayOperations.transpose_axes(self.matrix, (1, 0)).to_python(), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(ArrayOperations.take(self.matrix, 2).to_python(), [[1, 2], [4, 5]])
        self.assertEqual(ArrayOperations.take(self.matrix, 4).to_python(), [[1, 2, 3, 0], [4, 5, 6, 0]])
        self.assertEqual(ArrayOperations.take(self.matrix, 1, axis=0).to_python(), [[1, 2, 3]])
        self.assertEqual(ArrayOperations.drop(self.matrix, 1).to_python(), [[2, 3], [5, 6]])
        self.assertEqual(ArrayOperations.drop(self.matrix, 1, axis=0).to_python(), [[4, 5, 6]])
        self.assertEqual(ArrayOperations.first(self.matrix).to_python(), [1, 4])
        self.assertEqual(ArrayOperations.split(self.vector).to_python(), [2, 3, 4])
        self.assertEqual(
            ArrayOperations.catenate(ArrayType([1, 2], (1, 2)), ArrayType([3, 4], (1, 2))).to_python(),
            [[1, 2, 3, 4]],
        )
        self.assertEqual(
            ArrayOperations.catenate(ArrayType([1, 2], (1, 2)), ArrayType([3, 4], (1, 2)), axis=0).to_python(),
            [[1, 2], [3, 4]],
        )

    def test_reduce_and_scan_with_axis_support(self):
        self.assertEqual(ArrayOperations.sum(self.vector).to_python(), 10)
        self.assertEqual(ArrayOperations.sum(self.matrix).to_python(), [6, 15])
        self.assertEqual(ArrayOperations.sum(self.matrix, axis=0).to_python(), [5, 7, 9])
        self.assertEqual(ArrayOperations.product(self.matrix).to_python(), [6, 120])
        self.assertEqual(ArrayOperations.maximum(self.vector).to_python(), 4)
        self.assertEqual(ArrayOperations.minimum(self.vector).to_python(), 1)
        self.assertEqual(ArrayOperations.scan(MathOperations.add, self.matrix).to_python(), [[1, 3, 6], [4, 9, 15]])
        self.assertEqual(ArrayOperations.scan_first_axis(MathOperations.add, self.matrix).to_python(), [[1, 2, 3], [5, 7, 9]])

    def test_membership_partition_pick_and_disclose(self):
        self.assertEqual(ArrayOperations.membership(ArrayType([1, 2, 5]), ArrayType([2, 4, 5])).to_python(), [False, True, True])
        self.assertEqual(ArrayOperations.indices_where(ArrayType([0, 1, 0, 1, 1])).to_python(), [1, 3, 4])
        enclosed = ArrayOperations.enclose(self.vector)
        self.assertEqual(ArrayOperations.disclose(enclosed).to_python(), [1, 2, 3, 4])
        partitioned = ArrayOperations.partition(ArrayType([1, 0, 1, 0]), ArrayType([10, 20, 30, 40]))
        self.assertEqual([item.to_python() for item in partitioned.elements], [[10, 20], [30, 40]])
        self.assertEqual(ArrayOperations.pick(2, self.vector).to_python(), 3)
        self.assertEqual(ArrayOperations.pick(ArrayType([1, 2]), self.matrix).to_python(), 6)

    def test_grade_and_sorting(self):
        vector = ArrayType([30, 10, 20], (3,))
        self.assertEqual(ArrayOperations.grade_up(vector).to_python(), [1, 2, 0])
        self.assertEqual(ArrayOperations.grade_down(vector).to_python(), [0, 2, 1])
        keys = ArrayType([2, 3, 1], (3,))
        values = ArrayType(["b", "c", "a"], (3,))
        self.assertEqual(ArrayOperations.sort_by(keys, values).to_python(), ["a", "b", "c"])
        self.assertEqual(ArrayOperations.reverse_sort_by(keys, values).to_python(), ["c", "b", "a"])

    def test_error_handling(self):
        with self.assertRaises(ArithmeticError):
            MathOperations.divide(1, 0)
        with self.assertRaises(ValueError):
            ArrayOperations.reduce(MathOperations.add, ArrayType([], (0,)))
        with self.assertRaises(ValueError):
            ArrayOperations.catenate(ArrayType([1, 2], (1, 2)), ArrayType([1, 2, 3], (3, 1)))


if __name__ == "__main__":
    unittest.main()
