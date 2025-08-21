"""
Helpers to bridge dataclasses and msgspec.Struct
"""
import dataclasses
import msgspec
import typing
from typing import get_origin, get_args
from typing import get_type_hints, List, Dict, Type, Union, Any


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
        >>> decoded = reg.from_bytes(data, User)
        >>> decoded.tags
        ['x']
    """

    def __init__(self):
        self.cache: Dict[Type, Type] = {}  # dataclass -> struct

    def register(self, dc_cls: Type) -> Type[msgspec.Struct]:
        """Convert dataclass into msgspec.Struct (recursively)."""
        if dc_cls in self.cache:
            return self.cache[dc_cls]
        return dataclass_to_struct(dc_cls, self.cache)

    def to_dataclass(self, obj: Any, target_cls: Type = None) -> Any:
        """Recursively convert msgspec.Structs back to dataclasses."""
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
        if dataclasses.is_dataclass(origin):
            if isinstance(obj, origin):
                return obj
            # reconstruct dataclass
            field_values = {}
            hints = get_type_hints(origin, include_extras=True)
            for f in dataclasses.fields(origin):
                val = getattr(obj, f.name, None)
                field_values[f.name] = self.to_dataclass(val, hints.get(f.name))
            return origin(**field_values)

        # Handle List[T]
        if isinstance(obj, list) and origin in (list, List):
            subtype = get_args(target_cls)[0] if get_args(target_cls) else Any
            return [self.to_dataclass(v, subtype) for v in obj]

        # Handle Dict[K, V]
        if isinstance(obj, dict) and origin in (dict, Dict):
            k_type, v_type = get_args(target_cls) if get_args(target_cls) else (Any, Any)
            return {self.to_dataclass(k, k_type): self.to_dataclass(v, v_type)
                    for k, v in obj.items()}

        # Fallback
        return obj

    def decode(self, data: bytes, cls) -> Any:
        """Decode JSON bytes into the original dataclass via msgspec."""
        decoder = msgspec.json.Decoder(cls)
        struct_obj = decoder.decode(data)
        return struct_obj

    def from_bytes(self, data: bytes, dc_cls: Type) -> Any:
        """Decode JSON bytes into the original dataclass via msgspec."""
        cls = self.register(dc_cls)
        decoder = self.decode(data, cls)
        struct_obj = decoder.decode(data)
        return struct_obj


def dataclass_to_struct(
    dc_cls: Type,
    cache: Dict[Type, Type] = None
) -> Type[msgspec.Struct]:
    """
    Recursively convert a dataclass into a msgspec.Struct, handling nested
    dataclasses inside Optional, Union, List, Dict, etc.

    - Preserves defaults and default_factory
    - If a field is Optional[...] with no default, assigns = None
    - Uses kw_only=True to avoid required/optional reordering issues

    Example:
        >>> from magnet.utils.util_msgspec import *  # NOQA
        >>> import dataclasses, typing, msgspec
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
        >>> cache = {}
        >>> UserStruct = dataclass_to_struct(User, cache)
        >>> decoder = msgspec.json.Decoder(UserStruct)
        >>> data = b'{"id": 1, "name": "Alice", "address": {"city": "Paris", "zipcode": "75000"}, "tags": ["a", "b"]}'
        >>> obj = decoder.decode(data)
        >>> isinstance(obj, UserStruct)
        True
        >>> obj.id, obj.name, obj.address.city, obj.address.zipcode, obj.profile, obj.tags
        (1, 'Alice', 'Paris', '75000', None, ['a', 'b'])
    """
    if cache is None:
        cache = {}

    if not dataclasses.is_dataclass(dc_cls):
        raise TypeError(f"{dc_cls} is not a dataclass")

    if dc_cls in cache:
        return cache[dc_cls]

    hints = get_type_hints(dc_cls, include_extras=True)
    annotations = {}
    namespace = {}

    def convert_type(tp):
        """Recursively convert dataclass types inside annotations."""
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)

        # Direct dataclass
        if dataclasses.is_dataclass(tp):
            return dataclass_to_struct(tp, cache)

        # Optional[T] / Union[T, None]
        if origin is Union:
            new_args = tuple(convert_type(a) for a in args)
            return Union[new_args]  # rebuild Union

        # List[T]
        if origin in (list, List):
            return List[convert_type(args[0])]

        # Dict[K, V]
        if origin in (dict, Dict):
            k, v = args
            return Dict[convert_type(k), convert_type(v)]

        return tp

    for field in dataclasses.fields(dc_cls):
        field_type = convert_type(hints.get(field.name, field.type))
        annotations[field.name] = field_type

        if field.default is not dataclasses.MISSING:
            namespace[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore
            namespace[field.name] = dataclasses.field(default_factory=field.default_factory)
        else:
            # Special case: Optional[...] with no default -> assign None
            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)
            if origin is Union and type(None) in args:
                namespace[field.name] = None

    namespace['__annotations__'] = annotations
    namespace['__kw_only__'] = True  # allow mixed required/optional order

    struct_cls = type(dc_cls.__name__, (msgspec.Struct,), namespace, kw_only=True)
    cache[dc_cls] = struct_cls
    return struct_cls


MSGSPEC_REGISTRY = MsgspecRegistry()
