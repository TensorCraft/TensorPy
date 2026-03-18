BoolType = "bool"
NoneType = "NoneType"
NumberType = "float"
StringType = "str"
ListType = "list"
DictType = "dict"
TupleType = "tuple"
BytesType = "bytes"
FunctionType = "function"
BuiltinFunctionType = "builtin_function_or_method"
ModuleType = "module"
ClassType = "type"
TensorType = "tensor"
DeviceType = "device"
DTypeType = "dtype"


def type_name(value):
    return type(value)


def is_number(value):
    return type(value) == NumberType


def is_string(value):
    return type(value) == StringType


def is_list(value):
    return type(value) == ListType


def is_dict(value):
    return type(value) == DictType


def is_tuple(value):
    return type(value) == TupleType


def is_bytes(value):
    return type(value) == BytesType


def is_module(value):
    return type(value) == ModuleType


def is_tensor(value):
    return type(value) == TensorType


def is_device(value):
    return type(value) == DeviceType


def is_dtype(value):
    return type(value) == DTypeType
