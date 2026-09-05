import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from apl2_array_ops import ArrayOperations
from apl2_context import APLRuntime, APLContext, create_context, current_context, pop_context, push_context
from apl2_operations import MathOperations
from apl2_types import ArrayType, FloatingPointType, IntegerType


class TestAPLContext(unittest.TestCase):
    def tearDown(self):
        APLRuntime.destroy()

    def test_context_creation_defaults(self):
        context = create_context()
        self.assertEqual(context.index_origin, 0)
        self.assertEqual(context.print_width, 80)
        self.assertEqual(context.print_precision, 6)
        self.assertEqual(context.comparison_tolerance, 1e-15)

    def test_context_push_pop_nesting(self):
        first = APLContext(index_origin=1)
        second = APLContext(index_origin=0, print_precision=3)
        push_context(first)
        self.assertEqual(current_context().index_origin, 1)
        push_context(second)
        self.assertEqual(current_context().print_precision, 3)
        self.assertEqual(pop_context(), second)
        self.assertEqual(current_context(), first)
        self.assertEqual(pop_context(), first)
        self.assertEqual(current_context(), APLContext())

    def test_index_origin_affects_array_access(self):
        array = ArrayType([10, 20, 30], (3,))
        self.assertEqual(array.get((0,)).to_python(), 10)
        push_context(APLContext(index_origin=1))
        self.assertEqual(array.get((1,)).to_python(), 10)
        self.assertEqual(ArrayOperations.pick(IntegerType(2), array).to_python(), 20)
        with self.assertRaises(IndexError):
            array.get((0,))

    def test_print_precision_affects_formatting(self):
        push_context(APLContext(print_precision=3))
        self.assertEqual(MathOperations.format(FloatingPointType(3.1415926)).to_python(), "3.14")

    def test_print_width_affects_array_display_truncation(self):
        push_context(APLContext(print_width=12, print_precision=6))
        formatted = MathOperations.format(ArrayType([1, 2, 3, 4, 5], (5,))).to_python()
        self.assertEqual(formatted, "[1, 2, 3,...")

    def test_comparison_tolerance_affects_equality(self):
        self.assertFalse(MathOperations.equal(1.0, 1.0 + 1e-12).to_boolean())
        push_context(APLContext(comparison_tolerance=1e-9))
        self.assertTrue(MathOperations.equal(1.0, 1.0 + 1e-12).to_boolean())

    def test_async_context_isolation(self):
        async def read_with_origin(origin):
            push_context(APLContext(index_origin=origin))
            await asyncio.sleep(0)
            value = current_context().index_origin
            pop_context()
            return value

        async def run_tasks():
            return await asyncio.gather(read_with_origin(0), read_with_origin(1))

        values = asyncio.run(run_tasks())
        self.assertEqual(values, [0, 1])


if __name__ == "__main__":
    unittest.main()
