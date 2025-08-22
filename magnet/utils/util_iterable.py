"""
Helpers for iterables

Example:
    >>> from magnet.utils.util_iterable import *  # NOQA
    >>> from operator import length_hint
    >>> #
    >>> # Iterator case (works with anything iterable)
    >>> it = IteratorWithLength(iter([1, 2, 3, 4]), 4)
    >>> assert length_hint(it) == 4
    >>> next(it)
    >>> assert len(it) == 3
    >>> #
    >>> # Generator case (preserves send/throw/close)
    >>> def mygen():
    >>>     yield 10
    >>>     yield 20
    >>> #
    >>> gen = GeneratorWithLength(mygen(), 2)
    >>> assert length_hint(gen) == 2
    >>> assert next(gen) == 10
    >>> assert len(gen) == 1

Example:
    >>> from operator import length_hint
    >>> from magnet.utils.util_iterable import *  # NOQA
    >>> iterable = (_ for _ in range(10))
    >>> # Normally you cannot get the length of a generator
    >>> # the length hint will be zero
    >>> assert length_hint(iterable) == 0
    >>> # But if you know what it is, you can annotate it
    >>> self = IteratorWithLength(iterable, 10)
    >>> assert length_hint(self) == 10
    >>> next(self)
    >>> assert length_hint(self) == 9

Example:
    >>> from magnet.utils.util_iterable import *  # NOQA
    >>> iterable = (_ for _ in range(10))
    >>> # Normally you cannot get the length of a generator
    >>> # the length hint will be zero
    >>> try:
    >>>     len(iterable)
    >>> except TypeError:
    >>>     ...
    >>> else:
    >>>     raise AssertionError('unreachable')
    >>> # But if you know what it is, you can annotate it
    >>> self = IteratorWithLength(iterable, 10)
    >>> assert len(self) == 10
    >>> next(self)
    >>> assert len(self) == 9
"""
from collections.abc import Iterator, Generator, Sized
from typing import TypeVar, Generic, Optional
from typing import Union
from typing import overload
from typing import Literal
from typing import Any


T = TypeVar("T")


# --- Mixins for length logic ---

class _LengthHintMixin(Generic[T]):
    _remaining: int

    def __length_hint__(self) -> int:
        return max(0, self._remaining)

    def __str__(self):
        return str(self._wrapped)

    def __repr__(self):
        return f"<{self.__class__.__name__} remaining={self._remaining} obj={self._wrapped!r}>"


class _LengthMixin(_LengthHintMixin[T], Sized):
    def __len__(self) -> int:
        return self.__length_hint__()


# --- Iterator wrappers ---

class IteratorWithLengthHint(_LengthHintMixin[T], Iterator[T], Generic[T]):
    def __init__(self, it: Iterator[T], length_hint: int):
        self._wrapped = it
        self._remaining = length_hint

    def __iter__(self):
        return self

    def __next__(self) -> T:
        try:
            value = next(self._wrapped)
        except StopIteration:
            self._remaining = 0
            raise
        else:
            self._remaining -= 1
            return value


class IteratorWithLength(_LengthMixin[T], IteratorWithLengthHint[T]):
    """Adds __len__ on top of IteratorWithLengthHint"""


# --- Generator wrappers ---

class GeneratorWithLengthHint(_LengthHintMixin[T], Generator[T, Any, None], Generic[T]):
    def __init__(self, gen: Generator[T, Any, None], length_hint: int):
        self._wrapped = gen
        self._remaining = length_hint

    def __iter__(self):
        return self

    def __next__(self) -> T:
        try:
            value = next(self._wrapped)
        except StopIteration:
            self._remaining = 0
            raise
        else:
            self._remaining -= 1
            return value

    def send(self, value: Optional[object]) -> T:
        try:
            result = self._wrapped.send(value)
        except StopIteration:
            self._remaining = 0
            raise
        else:
            self._remaining -= 1
            return result

    def throw(self, typ, val=None, tb=None):
        try:
            return self._wrapped.throw(typ, val, tb)
        except StopIteration:
            self._remaining = 0
            raise

    def close(self):
        try:
            return self._wrapped.close()
        finally:
            self._remaining = 0


class GeneratorWithLength(_LengthMixin[T], GeneratorWithLengthHint[T]):
    """Adds __len__ on top of GeneratorWithLengthHint"""


# --- Factory helper ---

# Return aliases
LengthWrapped = Union[
    IteratorWithLengthHint[T],
    IteratorWithLength[T],
    GeneratorWithLengthHint[T],
    GeneratorWithLength[T],
]
StrictWrap = Union[IteratorWithLength[T], GeneratorWithLength[T]]
HintWrap   = Union[IteratorWithLengthHint[T], GeneratorWithLengthHint[T]]


@overload
def add_length_hint(obj: Iterator[T], length: int, known_length: Literal[True]) -> StrictWrap[T]:
    ...


@overload
def add_length_hint(obj: Iterator[T], length: int, known_length: Literal[False] = ...) -> HintWrap[T]:
    ...


def add_length_hint(
    obj: Union[Iterator[T], Generator[T, Any, None]],
    length: int,
    known_length: bool = False
) -> LengthWrapped[T]:
    """
    Wraps an iterator or generator to provide __length_hint__ (and optionally __len__).

    Args:
        obj: The iterator or generator to wrap.
        length: The initial length hint or known length.
        known_length: If True, returns a wrapper with __len__.
                If False, only __length_hint__ is available.

    Returns:
        A wrapped iterator/generator with length tracking.

    Example:
        >>> from magnet.utils.util_iterable import *  # NOQA
        >>> from operator import length_hint
        >>> # ----------------
        >>> # Generic iterator
        >>> it = add_length_hint(iter([1,2,3,4]), 4, known_length=False)
        >>> length_hint(it)
        4
        >>> next(it)
        1
        >>> length_hint(it)
        3
        >>> # ---------------------------
        >>> # Iterator with known length
        >>> it_strict = add_length_hint(iter([1,2,3,4]), 4, known_length=True)
        >>> len(it_strict)
        4
        >>> next(it_strict)
        1
        >>> len(it_strict)
        3
        >>> # ---------------------------
        >>> # Generator
        >>> def g():
        ...     yield from range(5)
        >>> gen = add_length_hint(g(), 5, known_length=False)
        >>> length_hint(gen)
        5
        >>> next(gen)
        0
        >>> length_hint(gen)
        4
        >>> # ---------------------------
        >>> # Generator with known length
        >>> gen_strict = add_length_hint(g(), 5, known_length=True)
        >>> len(gen_strict)
        5
        >>> next(gen_strict)
        0
        >>> len(gen_strict)
        4
    """
    if isinstance(obj, Generator):
        return GeneratorWithLength(obj, length) if known_length else GeneratorWithLengthHint(obj, length)
    elif isinstance(obj, Iterator):
        return IteratorWithLength(obj, length) if known_length else IteratorWithLengthHint(obj, length)
    else:
        raise TypeError(f"Object of type {type(obj)} is not an Iterator or Generator")
