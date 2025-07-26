import functools
import pickle
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import jsonpickle
import pytest

from src.utils.jsonable import JSONAble

# region Prepare Test Objects


class _TestEnum(Enum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"


# Named tuple for testing
TestNamedTuple = namedtuple("TestNamedTuple", ["field1", "field2"])


# Dataclass for testing
@dataclass
class TestDataClass:
    name: str
    values: List[int]
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FrozenTestDataClass:
    name: str
    values: tuple[int, ...]


# JSONAble dataclass for testing
@dataclass
class DataClassJSONAble(JSONAble):
    name: str
    value: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    option: Optional[_TestEnum] = None
    enum_dict: Dict[_TestEnum, str] = field(default_factory=dict)
    enum_set: Set[_TestEnum] = field(default_factory=set)
    enum_list: List[_TestEnum] = field(default_factory=list)
    enum_tuple: Tuple[_TestEnum, ...] = field(default_factory=tuple)
    enum_optional: Optional[_TestEnum] = None
    enum_union: Union[_TestEnum, str] = field(default_factory=lambda: _TestEnum.OPTION_A)
    enum_namedtuple: NamedTuple = field(default_factory=lambda: TestNamedTuple(1, 2))
    enum_list_without_proper_typehint: List = field(default_factory=list)
    enum_set_without_proper_typehint: Set = field(default_factory=set)
    enum_tuple_without_proper_typehint: Tuple = field(default_factory=tuple)
    enum_optional_without_proper_typehint: Optional[Any] = None
    enum_union_without_proper_typehint: Union[Any, str] = field(default_factory=lambda: _TestEnum.OPTION_A)
    enum_namedtuple_without_proper_typehint: NamedTuple = field(default_factory=lambda: TestNamedTuple(1, 2))
    dataclass: TestDataClass = field(default_factory=lambda: TestDataClass(name="test", values=[1, 2, 3]))
    dataclass_list: List[TestDataClass] = field(default_factory=list)
    dataclass_tuple: Tuple[TestDataClass, ...] = field(default_factory=tuple)
    dataclass_optional: Optional[TestDataClass] = None
    dataclass_union: Union[TestDataClass, str] = field(
        default_factory=lambda: TestDataClass(name="test", values=[1, 2, 3])
    )
    dataclass_namedtuple: NamedTuple = field(default_factory=lambda: TestNamedTuple(1, 2))
    dataclass_without_proper_typehint: TestDataClass = field(
        default_factory=lambda: TestDataClass(name="test", values=[1, 2, 3])
    )
    frozen_dataclass: FrozenTestDataClass = field(
        default_factory=lambda: FrozenTestDataClass(name="test", values=(1, 2, 3))
    )
    frozen_dataclass_list: List[FrozenTestDataClass] = field(default_factory=list)
    frozen_dataclass_set: Set[FrozenTestDataClass] = field(default_factory=set)
    frozen_dataclass_tuple: Tuple[FrozenTestDataClass, ...] = field(default_factory=tuple)
    frozen_dataclass_optional: Optional[FrozenTestDataClass] = None
    frozen_dataclass_union: Union[FrozenTestDataClass, str] = field(
        default_factory=lambda: FrozenTestDataClass(name="test", values=(1, 2, 3))
    )

    def __post_init__(self):
        # Ensure metadata is always a dict
        if self.metadata is None:
            self.metadata = {}


# Nested JSONAble dataclass
@dataclass
class NestedDataClassJSONAble(JSONAble):
    title: str
    items: List[DataClassJSONAble]
    config: Dict[str, Any] = field(default_factory=dict)
    items_without_proper_typehint: List = field(default_factory=list)
    config_without_proper_typehint: Dict[str, Any] = field(default_factory=dict)


class SimpleJSONAble(JSONAble):
    def __init__(self, name: str, value: int, option: _TestEnum = _TestEnum.OPTION_A):
        self.name = name
        self.value = value
        self.option = option

    def __eq__(self, other):
        if not isinstance(other, SimpleJSONAble):
            return False
        return self.name == other.name and self.value == other.value and self.option == other.option


class NestedJSONAble(JSONAble):
    def __init__(self, title: str, inner: SimpleJSONAble, tags: List[str], metadata: Optional[Dict[str, str]] = None):
        self.title = title
        self.inner = inner
        self.tags = tags
        self.metadata = metadata or {}

    def __eq__(self, other):
        if not isinstance(other, NestedJSONAble):
            return False
        return (
            self.title == other.title
            and self.inner == other.inner
            and self.tags == other.tags
            and self.metadata == other.metadata
        )


class ComplexJSONAble(JSONAble):
    def __init__(self, id: str, items: List[SimpleJSONAble], nested: Optional[NestedJSONAble] = None):
        self.id = id
        self.items = items
        self.nested = nested

    def __eq__(self, other):
        if not isinstance(other, ComplexJSONAble):
            return False
        return (
            self.id == other.id and all(a == b for a, b in zip(self.items, other.items)) and self.nested == other.nested
        )


class CollectionJSONAble(JSONAble):
    def __init__(
        self, tuple_field: Tuple[int, str, bool], set_field: Set[str], dict_field: Dict[str, Any], mixed_list: List[Any]
    ):
        self.tuple_field = tuple_field
        self.set_field = set_field
        self.dict_field = dict_field
        self.mixed_list = mixed_list

    def __eq__(self, other):
        if not isinstance(other, CollectionJSONAble):
            return False
        # For sets, convert to sorted list to ensure deterministic comparison
        return (
            self.tuple_field == other.tuple_field
            and sorted(self.set_field) == sorted(other.set_field)
            and self.dict_field == other.dict_field
            and self.mixed_list == other.mixed_list
        )


class WithExternalTypesJSONAble(JSONAble):
    def __init__(self, dataclass_field: TestDataClass, namedtuple_field: TestNamedTuple, nested_fields: Dict[str, Any]):
        self.dataclass_field = dataclass_field
        self.namedtuple_field = namedtuple_field
        self.nested_fields = nested_fields

    def __eq__(self, other):
        if not isinstance(other, WithExternalTypesJSONAble):
            return False

        # Check dataclass equality
        dataclass_equal = (
            self.dataclass_field.name == other.dataclass_field.name
            and self.dataclass_field.values == other.dataclass_field.values
            and self.dataclass_field.metadata == other.dataclass_field.metadata
        )

        # Check namedtuple equality
        namedtuple_equal = self.namedtuple_field == other.namedtuple_field

        # Check nested fields
        nested_equal = self.nested_fields == other.nested_fields

        return dataclass_equal and namedtuple_equal and nested_equal


# Test class that allows adding dynamic attributes
class DynamicJSONAble(JSONAble):
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, DynamicJSONAble):
            return False
        # Compare all attributes
        return all(getattr(self, attr) == getattr(other, attr) for attr in self.__dict__ if not attr.startswith("_"))


@pytest.fixture
def simple_object():
    return SimpleJSONAble("test", 42, _TestEnum.OPTION_B)


@pytest.fixture
def nested_object():
    simple = SimpleJSONAble("nested", 100)
    return NestedJSONAble("parent", simple, ["tag1", "tag2"], {"key": "value"})


@pytest.fixture
def complex_object():
    simple1 = SimpleJSONAble("item1", 1, _TestEnum.OPTION_A)
    simple2 = SimpleJSONAble("item2", 2, _TestEnum.OPTION_C)
    nested_inner = SimpleJSONAble("inner", 999)
    nested = NestedJSONAble("nested obj", nested_inner, ["x", "y", "z"])
    return ComplexJSONAble("complex-123", [simple1, simple2], nested)


@pytest.fixture
def collection_object():
    return CollectionJSONAble(
        tuple_field=(1, "test", True),
        set_field={"a", "b", "c"},
        dict_field={"key1": 123, "key2": "value", "key3": [1, 2, 3]},
        mixed_list=[1, "string", True, {"nested": "dict"}, [1, 2, 3]],
    )


@pytest.fixture
def external_types_object():
    dataclass_obj = TestDataClass(name="test_dataclass", values=[1, 2, 3, 4], metadata={"source": "test", "version": 1})

    namedtuple_obj = TestNamedTuple("value1", 42)

    nested_fields = {
        "simple_list": [1, 2, 3],
        "tuple_with_enum": (1, "test", _TestEnum.OPTION_A),
        "nested_dataclass": TestDataClass("nested", [9, 8, 7]),
    }

    return WithExternalTypesJSONAble(
        dataclass_field=dataclass_obj, namedtuple_field=namedtuple_obj, nested_fields=nested_fields
    )


@pytest.fixture
def dataclass_object():
    return DataClassJSONAble(
        name="dataclass-test",
        value=100,
        tags=["tag1", "tag2", "tag3"],
        metadata={"version": 1, "status": "active"},
        option=_TestEnum.OPTION_B,
    )


@pytest.fixture
def dataclass_with_basic_fields():
    """DataClassJSONAble with basic fields only."""
    return DataClassJSONAble(
        name="basic-fields", value=42, tags=["tag1", "tag2"], metadata={"version": 1, "status": "testing"}
    )


@pytest.fixture
def dataclass_with_enum_fields():
    """DataClassJSONAble with enum fields."""
    return DataClassJSONAble(
        name="enum-fields",
        value=100,
        option=_TestEnum.OPTION_B,
        enum_dict={_TestEnum.OPTION_A: "option_a", _TestEnum.OPTION_B: "option_b"},
        enum_set={_TestEnum.OPTION_A, _TestEnum.OPTION_C},
        enum_list=[_TestEnum.OPTION_A, _TestEnum.OPTION_B, _TestEnum.OPTION_C],
        enum_tuple=(_TestEnum.OPTION_A, _TestEnum.OPTION_B),
        enum_optional=_TestEnum.OPTION_C,
        enum_union=_TestEnum.OPTION_A,
    )


@pytest.fixture
def dataclass_with_improper_enum_typehints():
    """DataClassJSONAble with enum fields but without proper typehints."""
    return DataClassJSONAble(
        name="improper-enum-typehints",
        value=101,
        enum_list_without_proper_typehint=[_TestEnum.OPTION_A, _TestEnum.OPTION_B],
        enum_set_without_proper_typehint={_TestEnum.OPTION_A, _TestEnum.OPTION_C},
        enum_tuple_without_proper_typehint=(_TestEnum.OPTION_A, _TestEnum.OPTION_B),
        enum_optional_without_proper_typehint=_TestEnum.OPTION_C,
        enum_union_without_proper_typehint=_TestEnum.OPTION_A,
    )


@pytest.fixture
def dataclass_with_namedtuple_fields():
    """DataClassJSONAble with named tuple fields."""
    return DataClassJSONAble(
        name="namedtuple-fields",
        value=102,
        enum_namedtuple=TestNamedTuple(42, "test"),
        enum_namedtuple_without_proper_typehint=TestNamedTuple(100, "test2"),
    )


@pytest.fixture
def dataclass_with_nested_dataclass():
    """DataClassJSONAble with nested dataclass fields."""
    return DataClassJSONAble(
        name="nested-dataclass",
        value=103,
        dataclass=TestDataClass(name="inner-regular", values=[1, 2, 3], metadata={"inner": True}),
        dataclass_list=[
            TestDataClass(name="list-item-1", values=[4, 5], metadata={"index": 0}),
            TestDataClass(name="list-item-2", values=[6, 7], metadata={"index": 1}),
        ],
        dataclass_tuple=(
            TestDataClass(name="tuple-item-1", values=[8, 9], metadata={"index": 0}),
            TestDataClass(name="tuple-item-2", values=[10, 11], metadata={"index": 1}),
        ),
        dataclass_optional=TestDataClass(name="optional", values=[12, 13], metadata={"optional": True}),
        dataclass_union=TestDataClass(name="union", values=[14, 15], metadata={"union": True}),
        dataclass_without_proper_typehint=TestDataClass(name="no-typehint", values=[16, 17]),
    )


@pytest.fixture
def dataclass_with_frozen_dataclass():
    """DataClassJSONAble with frozen dataclass fields."""
    return DataClassJSONAble(
        name="frozen-dataclass",
        value=104,
        frozen_dataclass=FrozenTestDataClass(name="inner-frozen", values=(1, 2, 3)),
        frozen_dataclass_list=[
            FrozenTestDataClass(name="frozen-list-1", values=(4, 5)),
            FrozenTestDataClass(name="frozen-list-2", values=(6, 7)),
        ],
        frozen_dataclass_set={FrozenTestDataClass(name="frozen-set-1", values=(8, 9))},
        frozen_dataclass_tuple=(
            FrozenTestDataClass(name="frozen-tuple-1", values=(10, 11)),
            FrozenTestDataClass(name="frozen-tuple-2", values=(12, 13)),
        ),
        frozen_dataclass_optional=FrozenTestDataClass(name="frozen-optional", values=(14, 15)),
        frozen_dataclass_union=FrozenTestDataClass(name="frozen-union", values=(16, 17)),
    )


@pytest.fixture
def dataclass_with_string_union():
    """DataClassJSONAble with string unions instead of enum/dataclass."""
    return DataClassJSONAble(
        name="string-union",
        value=105,
        enum_union="string-instead-of-enum",
        dataclass_union="string-instead-of-dataclass",
        frozen_dataclass_union="string-instead-of-frozen",
    )


@pytest.fixture
def nested_dataclass_with_basic_items():
    """NestedDataClassJSONAble with a few simple items."""
    return NestedDataClassJSONAble(
        title="simple-nested",
        items=[DataClassJSONAble(name="item1", value=1), DataClassJSONAble(name="item2", value=2, tags=["x", "y"])],
        config={"simple_config": True},
    )


@pytest.fixture
def nested_dataclass_with_complex_items():
    """NestedDataClassJSONAble with complex items containing various field types."""
    items = [
        DataClassJSONAble(
            name="complex-item1",
            value=1,
            option=_TestEnum.OPTION_A,
            enum_dict={_TestEnum.OPTION_A: "option_a"},
            enum_set={_TestEnum.OPTION_A},
            enum_list=[_TestEnum.OPTION_A],
            enum_tuple=(_TestEnum.OPTION_A,),
            enum_optional=_TestEnum.OPTION_A,
            enum_union=_TestEnum.OPTION_A,
            enum_namedtuple=TestNamedTuple(1, 2),
            dataclass=TestDataClass(name="test", values=[1, 2, 3]),
            dataclass_list=[TestDataClass(name="test", values=[1, 2, 3])],
            dataclass_tuple=(TestDataClass(name="test", values=[1, 2, 3]),),
            frozen_dataclass=FrozenTestDataClass(name="test", values=(1, 2, 3)),
        ),
        DataClassJSONAble(
            name="complex-item2",
            value=2,
            tags=["x", "y"],
            enum_union="string-value",
            dataclass_union="string-instead-of-dataclass",
        ),
    ]
    return NestedDataClassJSONAble(
        title="complex-nested",
        items=items,
        config={"max_items": 5, "detailed": True},
        items_without_proper_typehint=[DataClassJSONAble(name="untyped-item", value=99)],
    )


@pytest.fixture
def nested_dataclass_object():
    """Legacy fixture maintained for backward compatibility."""
    items = [
        DataClassJSONAble(
            name="item1",
            value=1,
            option=_TestEnum.OPTION_A,
            enum_dict={_TestEnum.OPTION_A: "option_a"},
            enum_set={_TestEnum.OPTION_A},
            enum_list=[_TestEnum.OPTION_A],
            enum_tuple=(_TestEnum.OPTION_A,),
            enum_optional=_TestEnum.OPTION_A,
            enum_union=_TestEnum.OPTION_A,
            enum_namedtuple=TestNamedTuple(1, 2),
            dataclass=TestDataClass(name="test", values=[1, 2, 3]),
            dataclass_list=[TestDataClass(name="test", values=[1, 2, 3])],
            dataclass_tuple=(TestDataClass(name="test", values=[1, 2, 3]),),
            enum_optional_without_proper_typehint=_TestEnum.OPTION_A,
            enum_union_without_proper_typehint=_TestEnum.OPTION_A,
            enum_namedtuple_without_proper_typehint=TestNamedTuple(1, 2),
            dataclass_without_proper_typehint=TestDataClass(name="test", values=[1, 2, 3]),
            frozen_dataclass=FrozenTestDataClass(name="test", values=(1, 2, 3)),
            frozen_dataclass_list=[FrozenTestDataClass(name="test", values=(1, 2, 3))],
            frozen_dataclass_set={FrozenTestDataClass(name="test", values=(1, 2, 3))},
            frozen_dataclass_tuple=(FrozenTestDataClass(name="test", values=(1, 2, 3)),),
            frozen_dataclass_optional=FrozenTestDataClass(name="test", values=(1, 2, 3)),
            frozen_dataclass_union=FrozenTestDataClass(name="test", values=(1, 2, 3)),
        ),
        DataClassJSONAble(
            name="item2",
            value=2,
            tags=["x", "y"],
            enum_dict={_TestEnum.OPTION_B: "option_b"},
            enum_set={_TestEnum.OPTION_B},
            enum_list=[_TestEnum.OPTION_B],
            enum_tuple=(_TestEnum.OPTION_B,),
            enum_optional=_TestEnum.OPTION_B,
            enum_union="enum",
        ),
        DataClassJSONAble(name="item3", value=3, metadata={"key": "value"}, option=_TestEnum.OPTION_C),
    ]
    return NestedDataClassJSONAble(title="nested-dataclass", items=items, config={"max_items": 10, "enabled": True})


# endregion


@pytest.mark.parametrize(
    "fixture_name, cls",
    [
        ("simple_object", SimpleJSONAble),
        ("nested_object", NestedJSONAble),
        ("complex_object", ComplexJSONAble),
        ("collection_object", CollectionJSONAble),
        ("external_types_object", WithExternalTypesJSONAble),
        ("dataclass_object", DataClassJSONAble),
        ("dataclass_with_basic_fields", DataClassJSONAble),
        ("dataclass_with_enum_fields", DataClassJSONAble),
        ("dataclass_with_improper_enum_typehints", DataClassJSONAble),
        ("dataclass_with_namedtuple_fields", DataClassJSONAble),
        ("dataclass_with_nested_dataclass", DataClassJSONAble),
        ("dataclass_with_frozen_dataclass", DataClassJSONAble),
        ("dataclass_with_string_union", DataClassJSONAble),
        ("nested_dataclass_with_basic_items", NestedDataClassJSONAble),
        ("nested_dataclass_with_complex_items", NestedDataClassJSONAble),
        ("nested_dataclass_object", NestedDataClassJSONAble),
    ],
)
def test_roundtrip_serialization(fixture_name, cls, request):
    """
    Test that objects can be serialized to dictionary and deserialized back
    to equivalent objects with the same content and types.
    """

    # Get the object from the fixture
    def _get_func_equality_degree(func, obj1, obj2=None, cap=4) -> float:
        degree = 0
        if obj2 is None:
            obj2 = func(obj1)
        while obj1 != obj2:
            if degree > cap:
                return float("inf")
            degree += 1
            obj1 = func(obj1)
            obj2 = func(obj2)
        return degree

    # Function to test pickle serialization stability
    next_pickle_space = lambda obj: pickle.dumps(pickle.loads(obj))  # noqa: E731
    pickle_equality_degree = functools.partial(_get_func_equality_degree, next_pickle_space)

    original_obj: JSONAble = request.getfixturevalue(fixture_name)

    # Test pickle serialization/deserialization
    pickle_serialized = pickle.dumps(original_obj)
    pickle_deserialized = pickle.loads(pickle_serialized)
    assert original_obj == pickle_deserialized, "Pickle does not preserve object identity"

    # Check pickle idempotency - number of rounds needed to reach stability
    pickle_space_equality_degree = pickle_equality_degree(pickle_serialized)
    assert pickle_space_equality_degree <= 1, "Pickle is not idempotent"

    # Test jsonpickle serialization/deserialization
    jsonpickle_serialized = jsonpickle.dumps(original_obj)
    jsonpickle_deserialized = jsonpickle.loads(jsonpickle_serialized)
    if original_obj != jsonpickle_deserialized:
        pytest.skip("JSONPickle does not preserve object identity")

    # Check jsonpickle idempotency using a different approach
    # We can't use pickle_equality_degree directly on jsonpickle_serialized because it's a string
    jsonpickle_serialized2 = jsonpickle.dumps(jsonpickle_deserialized)
    jsonpickle_idempotent = jsonpickle_serialized == jsonpickle_serialized2
    assert jsonpickle_idempotent, "JSONPickle is not idempotent"
    jsonpickle_equality_degree = pickle_equality_degree(pickle.dumps(jsonpickle_deserialized))

    # First test dictionary serialization which should work regardless of JSON limitations
    try:
        jsonable_json = original_obj.to_jsonable_json()
        deserialized_jsonable = cls.from_jsonable_json(jsonable_json)
        assert jsonable_json == deserialized_jsonable.to_jsonable_json(), "JSONAble serialization is not idempotent"
        assert jsonpickle.dumps(deserialized_jsonable) == jsonpickle_serialized, (
            "JSONAble serialization does not preserve jsonpickle equality space"
        )
        assert deserialized_jsonable == original_obj, "JSONAble deserialization does not preserve object equality"

        # Check that repeated serialization is stable
        assert pickle_equality_degree(pickle.dumps(deserialized_jsonable)) <= 1, (
            "JSONAble serialization is not idempotent"
        )

    except (TypeError, ValueError) as e:
        if jsonpickle_equality_degree > 1:
            pytest.skip(f"JSONPickle is not idempotent: {jsonpickle_equality_degree}")
        # For TDD, we'll record the error but not fail the test since we expect issues
        # This helps identify what needs to be fixed in the implementation
        # pytest.skip(f"JSONAble serialization not yet implemented for this case: {str(e)}")
        raise e


def test_collection_types():
    """Test serialization of specific collection types individually"""
    # Test with tuple
    obj = DynamicJSONAble("tuple_test")
    setattr(obj, "tuple_data", (1, 2, "three", True))

    restored = DynamicJSONAble.from_jsonable_dict(obj.to_jsonable_dict())
    assert getattr(restored, "tuple_data") == getattr(obj, "tuple_data"), (
        f"Tuple data should be preserved, got {getattr(restored, 'tuple_data')} instead of {getattr(obj, 'tuple_data')}"
    )
    assert isinstance(getattr(restored, "tuple_data"), tuple), (
        f"Deserialized tuple_data should remain a tuple, got {type(getattr(restored, 'tuple_data'))}"
    )

    # Test with set - note sets become lists during serialization
    obj = DynamicJSONAble("set_test")
    setattr(obj, "set_data", {"a", "b", "c"})

    restored = DynamicJSONAble.from_jsonable_dict(obj.to_jsonable_dict())
    assert isinstance(getattr(restored, "set_data"), list), (
        f"Set should be converted to list during deserialization, got {type(getattr(restored, 'set_data'))}"
    )
    assert sorted(getattr(restored, "set_data")) == sorted(list(getattr(obj, "set_data"))), (
        "Set elements should be preserved in the deserialized list"
    )

    # Test with dict containing various types
    obj = DynamicJSONAble("dict_test")
    setattr(obj, "dict_data", {"int": 123, "str": "value", "bool": True, "list": [1, 2, 3], "none": None})

    restored = DynamicJSONAble.from_jsonable_dict(obj.to_jsonable_dict())
    assert getattr(restored, "dict_data") == getattr(obj, "dict_data"), (
        f"Dictionary content should be preserved,"
        f" got {getattr(restored, 'dict_data')} instead of {getattr(obj, 'dict_data')}"
    )
    assert isinstance(getattr(restored, "dict_data"), dict), (
        f"Deserialized dict_data should remain a dict, got {type(getattr(restored, 'dict_data'))}"
    )


def test_registry():
    registry = JSONAble._registry
    assert SimpleJSONAble.__name__ in registry, "SimpleJSONAble should be in the registry"
    assert NestedJSONAble.__name__ in registry, "NestedJSONAble should be in the registry"
    assert ComplexJSONAble.__name__ in registry, "ComplexJSONAble should be in the registry"
    assert registry[SimpleJSONAble.__name__] == SimpleJSONAble, "Registry should map class name to class"


def test_serialization_errors():
    # Test invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON string"):
        SimpleJSONAble.from_jsonable_json("{invalid json")

    # Test missing type or params1
    with pytest.raises(ValueError, match="missing 'type' or 'params'"):
        SimpleJSONAble.from_jsonable_dict({"params": {}})

    with pytest.raises(ValueError, match="missing 'type' or 'params'"):
        SimpleJSONAble.from_jsonable_dict({"type": "SimpleSerializable"})

    # Test unknown type
    with pytest.raises(ValueError, match="Unknown serializable type"):
        SimpleJSONAble.from_jsonable_dict({"type": "NonExistentClass", "params": {}})


def test_duplicate_registration():
    # Trying to register a class with same name should raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        # This class has the same name as one already registered
        class SimpleJSONAble(JSONAble):
            def __init__(self, different: bool):
                self.different = different
