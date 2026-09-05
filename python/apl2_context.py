"""Runtime context support for APL2 operations."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class APLContext:
    index_origin: int = 0
    print_width: int = 80
    print_precision: int = 6
    comparison_tolerance: float = 1e-15


_DEFAULT_CONTEXT = APLContext()
_CONTEXT_STACK: ContextVar[tuple[APLContext, ...]] = ContextVar("_apl2_context_stack", default=())


class APLRuntime:
    """Context stack with async-safe storage."""

    @staticmethod
    def current() -> APLContext:
        stack = _CONTEXT_STACK.get()
        return stack[-1] if stack else _DEFAULT_CONTEXT

    @staticmethod
    def push(context: APLContext) -> None:
        if not isinstance(context, APLContext):
            raise TypeError("context must be an APLContext")
        stack = _CONTEXT_STACK.get()
        _CONTEXT_STACK.set(stack + (context,))

    @staticmethod
    def pop() -> APLContext:
        stack = _CONTEXT_STACK.get()
        if not stack:
            raise RuntimeError("No context to pop")
        _CONTEXT_STACK.set(stack[:-1])
        return stack[-1]

    @staticmethod
    def create(
        index_origin: int = 0,
        print_width: int = 80,
        print_precision: int = 6,
        comparison_tolerance: float = 1e-15,
    ) -> APLContext:
        return APLContext(
            index_origin=index_origin,
            print_width=print_width,
            print_precision=print_precision,
            comparison_tolerance=comparison_tolerance,
        )

    @staticmethod
    def destroy() -> None:
        _CONTEXT_STACK.set(())


def current_context() -> APLContext:
    return APLRuntime.current()


def push_context(context: APLContext) -> None:
    APLRuntime.push(context)


def pop_context() -> APLContext:
    return APLRuntime.pop()


def create_context(
    index_origin: int = 0,
    print_width: int = 80,
    print_precision: int = 6,
    comparison_tolerance: float = 1e-15,
) -> APLContext:
    return APLRuntime.create(
        index_origin=index_origin,
        print_width=print_width,
        print_precision=print_precision,
        comparison_tolerance=comparison_tolerance,
    )
