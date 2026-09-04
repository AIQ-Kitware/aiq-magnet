"""
Helpers to bridge dataclasses and msgspec.Struct
"""
from __future__ import annotations
import builtins
import dataclasses
import inspect
import ubelt as ub
import msgspec
import types
import typing
from typing import (
    Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints,
)


StructType = type[msgspec.Struct]
DataclassType = type[object]
T = TypeVar('T')


class MsgspecRegistry:
    """
    Registry that manages dataclass <-> msgspec.Struct mappings.

    Provides:
      - register(): convert dataclasses into Structs (recursively)
      - to_dataclass(): convert Struct instances back to dataclasses
      - from_text(): decode JSON bytes directly into original dataclasses

    Example:
        >>> from magnet.utils.util_msgspec import *  # NOQA
        >>> import dataclasses, typing
        >>> @dataclasses.dataclass
        ... class Address:
        ...     city: str
        ...     zipcode: str
        ...
        >>> @dataclasses.dataclass
        ... class Profile:
        ...     bio: typing.Optional[str] = None
        ...     website: typing.Optional[str] = None
        ...
        >>> @dataclasses.dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...     address: Address
        ...     profile: typing.Optional[Profile] = None
        ...     tags: typing.List[str] = dataclasses.field(default_factory=list)
        ...
        >>> reg = MsgspecRegistry()
        >>> UserStruct = reg.register(User)
        >>> #
        >>> # Decode JSON into msgspec struct
        >>> decoder = msgspec.json.Decoder(UserStruct)
        >>> data = b'{"id": 1, "name": "Alice", "address": {"city": "Paris", "zipcode": "75000"}, "tags": ["x"]}'
        >>> struct_obj = decoder.decode(data)
        >>> struct_obj
        User(id=1, name='Alice', address=Address(city='Paris', zipcode='75000'), profile=None, tags=['x'])
        >>> #
        >>> # Convert back to original dataclass
        >>> user = reg.to_dataclass(struct_obj)
        >>> user.address.city
        'Paris'
        >>> #
        >>> # Or directly from text
        >>> decoded = reg.decode(data, reg[User])
        >>> decoded.tags
        ['x']
    """

    def __init__(self) -> None:
        self.cache: dict[DataclassType, StructType] = {}
        self._hints_cache: dict[DataclassType, dict[str, Any]] = {}

    def __getitem__(self, key: DataclassType) -> StructType:
        return self.cache[key]

    def register(
        self,
        dc_cls: DataclassType,
        dict: bool = False,
        *,
        localns: typing.Mapping[str, Any] | None = None,
    ) -> StructType:
        """Convert dataclass into msgspec.Struct (recursively)."""
        if dc_cls in self.cache:
            return self.cache[dc_cls]
        if localns is None:
            localns = _caller_locals()
        return dataclass_to_struct(
            dc_cls,
            self.cache,
            dict=dict,
            localns=localns,
            hints_cache=self._hints_cache,
        )

    def to_dataclass(self, obj: Any, target_cls: Any | None = None) -> Any:
        """
        Recursively convert msgspec.Structs back to dataclasses.

        Note: this is fairly slow, and effectively removes the msgspec
        advantage.
        """
        if obj is None:
            return None

        # If this object is a registered struct, lookup the original dataclass
        if target_cls is None:
            for dc_cls, struct_cls in self.cache.items():
                if isinstance(obj, struct_cls):
                    target_cls = dc_cls
                    break

        if target_cls is None:
            return obj

        origin = get_origin(target_cls) or target_cls

        # Already correct type?
        if isinstance(origin, type) and dataclasses.is_dataclass(origin):
            dc_type = cast(type[Any], origin)
            if isinstance(obj, dc_type):
                return obj
            field_values: dict[str, Any] = {}
            hints = self._hints_cache.get(dc_type)
            if hints is None:
                hints = _resolve_type_hints(dc_type)
            for f in dataclasses.fields(cast(Any, dc_type)):
                val = getattr(obj, f.name, None)
                field_values[f.name] = self.to_dataclass(val, hints.get(f.name))
            return dc_type(**field_values)

        # Handle List[T]
        if isinstance(obj, list) and origin is list:
            subtype = get_args(target_cls)[0] if get_args(target_cls) else Any
            return [self.to_dataclass(v, subtype) for v in obj]

        # Handle Dict[K, V]
        if isinstance(obj, dict) and origin is dict:
            k_type, v_type = get_args(target_cls) if get_args(target_cls) else (Any, Any)
            return {self.to_dataclass(k, k_type): self.to_dataclass(v, v_type)
                    for k, v in obj.items()}

        # Fallback
        return obj

    def decode(self, data: bytes, cls: type[T]) -> T:
        """Decode JSON into one registered struct type."""
        decoder = msgspec.json.Decoder(cls)
        return decoder.decode(data)

    def decode_list(self, data: bytes, cls: type[T]) -> list[T]:
        """Decode JSON into a list of one registered struct type."""
        sequence_type = types.GenericAlias(list, cls)
        decoder = msgspec.json.Decoder(cast(Any, sequence_type))
        return cast(list[T], decoder.decode(data))

    # Broken
    # def from_bytes(self, data: bytes, dc_cls: Type) -> Any:
    #     """Decode JSON bytes into the original dataclass via msgspec."""
    #     cls = self.register(dc_cls)
    #     struct_obj = self.decode(data, cls)
    #     return struct_obj



def _caller_locals() -> dict[str, Any]:
    """Return a copy of the caller's caller local namespace."""
    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None:
            return {}
        caller = frame.f_back.f_back
        if caller is None:
            return {}
        return dict(caller.f_locals)
    finally:
        del frame


def _resolve_type_hints(
    dc_cls: DataclassType,
    localns: typing.Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve postponed annotations for module- or locally-defined classes."""
    resolved_localns = dict(localns or {})
    resolved_localns.setdefault(dc_cls.__name__, dc_cls)
    return get_type_hints(
        dc_cls,
        localns=resolved_localns,
        include_extras=True,
    )


def dataclass_to_struct(
    dc_cls: DataclassType,
    cache: dict[DataclassType, StructType] | None = None,
    dict: bool = False,
    *,
    localns: typing.Mapping[str, Any] | None = None,
    hints_cache: dict[DataclassType, dict[str, Any]] | None = None,
) -> StructType:
    """
    Recursively convert a dataclass into a msgspec.Struct, handling nested
    dataclasses inside Optional, Union, List, Dict, etc.

    - Preserves defaults and default_factory
    - If a field is Optional[...] with no default, assigns = None
    - Uses kw_only=True to avoid required/optional reordering issues

    Example:
        >>> from magnet.utils.util_msgspec import *  # NOQA
        >>> import dataclasses, typing, msgspec
        >>> @dataclasses.dataclass(eq=True, frozen=True)
        ... class Address:
        ...     city: str
        ...     zipcode: str
        ...
        >>> @dataclasses.dataclass
        ... class Profile:
        ...     bio: typing.Optional[str] = None
        ...     website: typing.Optional[str] = None
        ...
        >>> @dataclasses.dataclass
        ... class User:
        ...     id: int
        ...     name: str
        ...     address: Address
        ...     profile: typing.Optional[Profile] = None
        ...     tags: typing.List[str] = dataclasses.field(default_factory=list)
        ...
        >>> cache = {}
        >>> UserStruct = dataclass_to_struct(User, cache, dict=True)
        >>> decoder = msgspec.json.Decoder(UserStruct)
        >>> data = b'{"id": 1, "name": "Alice", "address": {"city": "Paris", "zipcode": "75000"}, "tags": ["a", "b"]}'
        >>> obj = decoder.decode(data)
        >>> obj.not_frozen = True  # we set frozen to false so this should work
        >>> isinstance(obj, UserStruct)
        True
        >>> obj.id, obj.name, obj.address.city, obj.address.zipcode, obj.profile, obj.tags
        (1, 'Alice', 'Paris', '75000', None, ['a', 'b'])
    """
    if cache is None:
        cache = {}
    if localns is None:
        localns = _caller_locals()

    if not dataclasses.is_dataclass(dc_cls):
        raise TypeError(f"{dc_cls} is not a dataclass")

    if dc_cls in cache:
        return cache[dc_cls]

    dparams = getattr(dc_cls, "__dataclass_params__", None)
    frozen = bool(getattr(dparams, "frozen", False))
    dc_eq = bool(getattr(dparams, "eq", True))

    hints = _resolve_type_hints(dc_cls, localns)
    if hints_cache is not None:
        hints_cache[dc_cls] = hints
    annotations: builtins.dict[str, Any] = {}
    namespace: builtins.dict[str, Any] = {}

    def convert_type(tp: Any) -> Any:
        """Recursively convert dataclass types inside annotations."""
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)

        # Direct dataclass
        if isinstance(tp, type) and dataclasses.is_dataclass(tp):
            return dataclass_to_struct(
                tp,
                cache,
                localns=localns,
                hints_cache=hints_cache,
            )

        # Optional[T] / Union[T, None]
        if origin is Union:
            new_args = tuple(convert_type(a) for a in args)
            return Union[new_args]  # rebuild Union

        # List[T]
        if origin is list:
            return types.GenericAlias(list, convert_type(args[0]))

        # Dict[K, V]
        if origin is builtins.dict:
            k, v = args
            args = (convert_type(k), convert_type(v))
            return types.GenericAlias(builtins.dict, args)

        return tp

    for field in dataclasses.fields(cast(Any, dc_cls)):
        field_type = convert_type(hints.get(field.name, field.type))
        annotations[field.name] = field_type

        if field.default is not dataclasses.MISSING:
            namespace[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            namespace[field.name] = dataclasses.field(default_factory=field.default_factory)
        else:
            # Special case: Optional[...] with no default -> assign None
            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)
            if origin is Union and type(None) in args:
                namespace[field.name] = None

    namespace['__annotations__'] = annotations
    namespace['__kw_only__'] = True  # allow mixed required/optional order

    struct_factory = cast(Any, type)
    struct_cls = cast(
        StructType,
        struct_factory(
            dc_cls.__name__,
            (msgspec.Struct,),
            namespace,
            kw_only=True,
            dict=dict,
            frozen=frozen,
            eq=dc_eq,
        ),
    )
    cache[dc_cls] = struct_cls
    return struct_cls


@ub.hash_data.register(msgspec.Struct)  # ty: ignore[unresolved-attribute]
def _hash_msgspec(data: msgspec.Struct):
    """
    Dataclasses don't dispatch.

    Example:
        >>> from magnet.utils.util_msgspec import *  # NOQA
        >>> import msgspec
        >>> class P(msgspec.Struct):
        >>>     x: int
        >>>     y: int
        >>> #
        >>> a = P(1, 2)
        >>> b = P(1, 2)
        >>> c = P(2, 1)
        >>> #
        >>> assert ub.hash_data(a) == ub.hash_data(b)
        >>> assert ub.hash_data(a) != ub.hash_data(c)
        >>> # CHeck dataclass compat
        >>> import dataclasses
        >>> @dataclasses.dataclass
        >>> class P:
        >>>     x: int
        >>>     y: int
        >>> #
        >>> a2 = P(1, 2)
        >>> b2 = P(1, 2)
        >>> c2 = P(2, 1)
        >>> #
        >>> print(ub.hash_data(a))
        >>> print(ub.hash_data(a2))
        >>> print(ub.util_hash._hashable_sequence(a))
        >>> print(ub.util_hash._hashable_sequence(a2))
    """
    from msgspec import structs
    from ubelt import util_hash

    cls = data.__class__
    header = (cls.__module__, cls.__qualname__)

    # fields() preserves the class' field definition order
    flds = structs.fields(cls)  # or structs.fields(data)
    items = [(f.name, getattr(data, f.name)) for f in flds]

    # Reuse ubelt's existing machinery to recurse into values
    seq = util_hash._hashable_sequence(
        (header, items),
        extensions=ub.hash_data.extensions,  # ty: ignore[unresolved-attribute]
        types=util_hash._COMPATIBLE_HASHABLE_SEQUENCE_TYPES_DEFAULT,
    )
    prefix = b'DCLASS'
    hashable = b''.join(seq)
    return prefix, hashable


MSGSPEC_REGISTRY = MsgspecRegistry()


def asdict(struct: msgspec.Struct) -> dict[str, Any]:
    """
    Mirror dataclasses.asdict
    """
    import msgspec
    import kwutil
    return kwutil.Json.loads(msgspec.json.encode(struct), backend='orjson')
