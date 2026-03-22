import math
import ml
import random
import io
import json


def _shape_product(shape):
    total = 1
    for dim in shape:
        total = total * dim
    return total


def _rand_uniform(shape, low, high):
    if len(shape) == 0:
        return low + (high - low) * random.random()

    out = []
    count = shape[0]
    child_shape = shape[1:]
    i = 0
    while i < count:
        out.append(_rand_uniform(child_shape, low, high))
        i = i + 1
    return out


def _zeros_like_shape(shape):
    if len(shape) == 0:
        return 0

    out = []
    i = 0
    while i < shape[0]:
        out.append(_zeros_like_shape(shape[1:]))
        i = i + 1
    return out


def _is_sequence(value):
    return type(value) == "list" or type(value) == "tuple"


def _is_tensor(value):
    return type(value) == "tensor"


def _is_module(value):
    return isinstance(value, Module)


def _is_state_container(value):
    if value == None:
        return False
    if _is_tensor(value) or _is_module(value):
        return True
    if _is_sequence(value):
        i = 0
        while i < len(value):
            if _is_state_container(value[i]):
                return True
            i = i + 1
        return False
    if type(value) == "dict":
        for key in value:
            if _is_state_container(value[key]):
                return True
        return False
    return False


def _flatten_values(value, out):
    if _is_sequence(value):
        for item in value:
            _flatten_values(item, out)
    else:
        out.append(value)


def _index_shape(value):
    if _is_sequence(value):
        if len(value) == 0:
            return [0]
        shape = [len(value)]
        child = _index_shape(value[0])
        i = 1
        while i < len(value):
            if not _same_shape(_index_shape(value[i]), child):
                raise ValueError("Embedding indices must form a rectangular shape.")
            i = i + 1
        shape.extend(child)
        return shape
    return []


def _same_shape(left, right):
    if len(left) != len(right):
        return False

    i = 0
    while i < len(left):
        if left[i] != right[i]:
            return False
        i = i + 1
    return True


def _sorted_keys(mapping):
    keys = []
    for key in mapping:
        keys.append(key)
    keys.sort()
    return keys


def _collect_parameters(value, out):
    if value == None:
        return

    if _is_tensor(value):
        if value.requires_grad:
            out.append(value)
        return

    if _is_module(value):
        for name in dir(value):
            if name.startswith("__"):
                continue
            _collect_parameters(getattr(value, name), out)
        return

    if _is_sequence(value):
        for item in value:
            _collect_parameters(item, out)
        return

    if type(value) == "dict":
        for key in value:
            _collect_parameters(value[key], out)


def _device_name(device):
    if device == None:
        return None
    return device.name


def _dtype_name(dtype):
    if dtype == None:
        return None
    return dtype.name


def _serialize_tensor(tensor):
    return {
        "kind": "tensor",
        "shape": tensor.shape,
        "dtype": _dtype_name(tensor.dtype),
        "device": _device_name(tensor.device),
        "requires_grad": tensor.requires_grad,
        "data": tensor.tolist(),
    }


def _serialize_state_value(value):
    if value == None:
        return None
    if _is_tensor(value):
        return _serialize_tensor(value)
    if _is_sequence(value):
        out = []
        i = 0
        while i < len(value):
            out.append(_serialize_state_value(value[i]))
            i = i + 1
        return out
    if type(value) == "dict":
        out = {}
        for key in _sorted_keys(value):
            out[str(key)] = _serialize_state_value(value[key])
        return out
    return value


def _target_device_name(saved_device, map_location):
    if map_location == None:
        return saved_device
    if type(map_location) == "device":
        return map_location.name
    return str(map_location)


def _validate_parameter_device(target_device, requires_grad):
    if requires_grad and target_device != None and target_device.name != "cpu":
        raise RuntimeError("Trainable module parameters currently support only CPU devices.")


def _deserialize_tensor(state, current=None, map_location=None):
    target_dtype = None
    target_device = None

    if current != None:
        target_dtype = current.dtype
        if map_location == None:
            target_device = current.device

    if target_dtype == None and state.get("dtype") != None:
        target_dtype = ml.dtype(state["dtype"])

    device_name = _target_device_name(state.get("device"), map_location)
    if target_device == None and device_name != None:
        target_device = ml.device(device_name)

    requires_grad = state.get("requires_grad")
    if current != None and current.requires_grad:
        requires_grad = True

    _validate_parameter_device(target_device, requires_grad)

    if requires_grad:
        return ml.Parameter(state.get("data"), target_dtype, target_device)
    return ml.tensor(state.get("data"), target_dtype, target_device)


def _deserialize_state_value(state, current=None, map_location=None):
    if state == None:
        return None

    if type(state) == "dict" and state.get("kind") == "tensor":
        return _deserialize_tensor(state, current, map_location)

    if type(state) == "list":
        out = []
        i = 0
        while i < len(state):
            current_item = None
            if type(current) == "list" and i < len(current):
                current_item = current[i]
            out.append(_deserialize_state_value(state[i], current_item, map_location))
            i = i + 1
        return out

    if type(state) == "dict":
        out = {}
        for key in _sorted_keys(state):
            current_item = None
            if type(current) == "dict" and key in current:
                current_item = current[key]
            out[key] = _deserialize_state_value(state[key], current_item, map_location)
        return out

    return state


def _collect_state(value, prefix, out):
    if value == None:
        return

    if _is_tensor(value):
        out[prefix] = _serialize_tensor(value)
        return

    if _is_module(value):
        for name in dir(value):
            if name.startswith("__"):
                continue
            child = getattr(value, name)
            if not _is_state_container(child):
                continue
            child_prefix = name
            if prefix != "":
                child_prefix = prefix + "." + name
            _collect_state(child, child_prefix, out)
        return

    if _is_sequence(value):
        i = 0
        while i < len(value):
            child = value[i]
            if _is_state_container(child):
                child_prefix = str(i)
                if prefix != "":
                    child_prefix = prefix + "." + child_prefix
                _collect_state(child, child_prefix, out)
            i = i + 1
        return

    if type(value) == "dict":
        for key in _sorted_keys(value):
            child = value[key]
            if _is_state_container(child):
                child_prefix = str(key)
                if prefix != "":
                    child_prefix = prefix + "." + child_prefix
                _collect_state(child, child_prefix, out)


def _load_state(value, prefix, state, map_location, missing, loaded):
    if value == None:
        return value

    if _is_tensor(value):
        if not (prefix in state):
            missing.append(prefix)
            return value
        loaded.append(prefix)
        return _deserialize_tensor(state[prefix], value, map_location)

    if _is_module(value):
        for name in dir(value):
            if name.startswith("__"):
                continue
            child = getattr(value, name)
            if not _is_state_container(child):
                continue
            child_prefix = name
            if prefix != "":
                child_prefix = prefix + "." + name
            updated = _load_state(child, child_prefix, state, map_location, missing, loaded)
            if updated != child:
                setattr(value, name, updated)
        return value

    if type(value) == "list":
        i = 0
        while i < len(value):
            child = value[i]
            if _is_state_container(child):
                child_prefix = str(i)
                if prefix != "":
                    child_prefix = prefix + "." + child_prefix
                value[i] = _load_state(child, child_prefix, state, map_location, missing, loaded)
            i = i + 1
        return value

    if type(value) == "tuple":
        raise ValueError("Tuple-backed state containers are not supported.")

    if type(value) == "dict":
        for key in value:
            child = value[key]
            if _is_state_container(child):
                child_prefix = str(key)
                if prefix != "":
                    child_prefix = prefix + "." + child_prefix
                value[key] = _load_state(child, child_prefix, state, map_location, missing, loaded)
        return value

    return value


def _move_state_value(value, device):
    if value == None:
        return value

    if _is_tensor(value):
        target_device = device
        if type(device) != "device":
            target_device = ml.device(str(device))
        _validate_parameter_device(target_device, value.requires_grad)
        moved = value.to(device)
        if value.requires_grad:
            return ml.Parameter(moved.tolist(), moved.dtype, moved.device)
        return moved

    if _is_module(value):
        for name in dir(value):
            if name.startswith("__"):
                continue
            child = getattr(value, name)
            if not _is_state_container(child):
                continue
            updated = _move_state_value(child, device)
            if updated != child:
                setattr(value, name, updated)
        return value

    if type(value) == "list":
        i = 0
        while i < len(value):
            child = value[i]
            if _is_state_container(child):
                value[i] = _move_state_value(child, device)
            i = i + 1
        return value

    if type(value) == "tuple":
        raise ValueError("Tuple-backed state containers are not supported.")

    if type(value) == "dict":
        for key in value:
            child = value[key]
            if _is_state_container(child):
                value[key] = _move_state_value(child, device)
        return value

    return value


def _normalize_indices(indices):
    if _is_tensor(indices):
        return indices.tolist()
    return indices


def _one_hot_rows(flat_indices, depth):
    rows = []
    i = 0
    while i < len(flat_indices):
        index = int(flat_indices[i])
        if index < 0 or index >= depth:
            raise IndexError("Embedding index out of range.")
        row = []
        j = 0
        while j < depth:
            if j == index:
                row.append(1)
            else:
                row.append(0)
            j = j + 1
        rows.append(row)
        i = i + 1
    return rows


def _tensor_step_input(sequence_data, step):
    if type(sequence_data) != "list":
        raise ValueError("Expected sequence data to be a list.")

    if len(sequence_data) == 0:
        return []

    if type(sequence_data[0]) == "list" and len(sequence_data[0]) > 0 and type(sequence_data[0][0]) == "list":
        batch = []
        i = 0
        while i < len(sequence_data):
            batch.append(sequence_data[i][step])
            i = i + 1
        return batch

    return sequence_data[step]


def _sequence_length(sequence_data):
    if type(sequence_data) != "list":
        return 0
    if len(sequence_data) == 0:
        return 0
    if type(sequence_data[0]) == "list" and len(sequence_data[0]) > 0 and type(sequence_data[0][0]) == "list":
        return len(sequence_data[0])
    return len(sequence_data)


def _to_batch_first_data(x, batch_first):
    data = _normalize_indices(x)
    if _is_tensor(x):
        data = x.tolist()
    if not batch_first:
        return _transpose_time_batch(data)
    return data


def _transpose_time_batch(data):
    if type(data) != "list" or len(data) == 0:
        return data

    batch = len(data)
    steps = len(data[0])
    out = []
    step = 0
    while step < steps:
        row = []
        item = 0
        while item < batch:
            row.append(data[item][step])
            item = item + 1
        out.append(row)
        step = step + 1
    return out


def _stack_tensor_list(items):
    values = []
    for item in items:
        values.append(item.tolist())
    return ml.tensor(values)


def _as_state_pair(state):
    if state == None:
        return None
    if type(state) == "tuple" or type(state) == "list":
        return [state[0], state[1]]
    raise ValueError("Expected recurrent state pair.")


def _init_hidden_like_input(x, hidden_size):
    if x.rank == 1:
        return ml.zeros([hidden_size])
    return ml.zeros([x.shape[0], hidden_size])


def _expand_hidden_for_return(hidden):
    if hidden.rank == 1:
        return ml.reshape(hidden, [1, hidden.shape[0]])
    return ml.reshape(hidden, [1, hidden.shape[0], hidden.shape[1]])


def _run_recurrent_sequence(cell, x, initial_state, batch_first):
    sequence_data = _to_batch_first_data(x, batch_first)
    steps = _sequence_length(sequence_data)
    outputs = []
    state = initial_state

    step = 0
    while step < steps:
        step_input = _tensor_step_input(sequence_data, step)
        step_tensor = ml.tensor(step_input)
        state = cell.forward(step_tensor, state)
        if type(state) == "list" or type(state) == "tuple":
            outputs.append(state[0])
        else:
            outputs.append(state)
        step = step + 1

    if len(outputs) == 0:
        return [ml.tensor([]), state]

    output = _stack_tensor_list(outputs)
    if batch_first:
        output = ml.tensor(_transpose_time_batch(output.tolist()))
    return [output, state]


class Module:

    def state_dict(self):
        out = {}
        _collect_state(self, "", out)
        return out

    def load_state_dict(self, state, map_location=None, strict=True):
        missing = []
        loaded = []
        _load_state(self, "", state, map_location, missing, loaded)

        unexpected = []
        for key in _sorted_keys(state):
            if key not in loaded:
                unexpected.append(key)

        if strict and len(missing) > 0:
            raise ValueError("Missing state keys: " + ", ".join(missing))
        if strict and len(unexpected) > 0:
            raise ValueError("Unexpected state keys: " + ", ".join(unexpected))
        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    def save(self, path):
        payload = {
            "format": "tensorpy.nn.state_dict.v1",
            "state": self.state_dict(),
        }
        io.write_text(path, json.dumps(payload))
        return path

    def load(self, path, map_location=None, strict=True):
        payload = json.loads(io.read_text(path))
        if type(payload) != "dict" or payload.get("format") != "tensorpy.nn.state_dict.v1":
            raise ValueError("Invalid TensorPy model checkpoint.")
        return self.load_state_dict(payload.get("state"), map_location, strict)

    def to(self, device):
        _move_state_value(self, device)
        return self

    def parameters(self):
        params = []
        _collect_parameters(self, params)
        return params

    def children(self):
        out = []
        for name in dir(self):
            if name.startswith("__"):
                continue
            value = getattr(self, name)
            if _is_module(value):
                out.append(value)
            elif _is_sequence(value):
                for item in value:
                    if _is_module(item):
                        out.append(item)
        return out

    def zero_grad(self):
        ml.zero_grad(self.parameters())


class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        limit = 1.0 / math.sqrt(in_features)
        self.weight = ml.Parameter(_rand_uniform([in_features, out_features], -limit, limit))
        if bias:
            self.bias = ml.Parameter(_rand_uniform([out_features], -limit, limit))
        else:
            self.bias = None

    def forward(self, x):
        out = ml.matmul(x, self.weight)
        if self.bias != None:
            out = ml.add(out, self.bias)
        return out


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        limit = 1.0 / math.sqrt(embedding_dim)
        self.weight = ml.Parameter(_rand_uniform([num_embeddings, embedding_dim], -limit, limit))

    def forward(self, indices):
        normalized = _normalize_indices(indices)
        shape = _index_shape(normalized)
        flat = []
        _flatten_values(normalized, flat)

        if len(shape) == 0:
            one_hot = ml.tensor(_one_hot_rows([flat[0]], self.num_embeddings)[0])
            return ml.matmul(one_hot, self.weight)

        rows = _one_hot_rows(flat, self.num_embeddings)
        encoded = ml.tensor(rows)
        out = ml.matmul(encoded, self.weight)
        out_shape = []
        for dim in shape:
            out_shape.append(dim)
        out_shape.append(self.embedding_dim)
        return ml.reshape(out, out_shape)


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        if isinstance(kernel_size, "tuple") or isinstance(kernel_size, "list"):
            kh = kernel_size[0]
            kw = kernel_size[1]
        else:
            kh = kernel_size
            kw = kernel_size

        scale = 1.0 / math.sqrt(in_channels * kh * kw)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.weight = ml.Parameter(_rand_uniform([out_channels, in_channels, kh, kw], -scale, scale))
        if bias:
            self.bias = ml.Parameter(_rand_uniform([1, out_channels, 1, 1], -scale, scale))
        else:
            self.bias = None

    def forward(self, x):
        out = ml.conv2d(x, self.weight)
        if self.bias != None:
            out = ml.add(out, self.bias)
        return out


class ReLU(Module):

    def forward(self, x):
        return ml.relu(x)


class Tanh(Module):

    def forward(self, x):
        return ml.tanh(x)


class Sigmoid(Module):

    def forward(self, x):
        return ml.sigmoid(x)


class Flatten(Module):

    def forward(self, x):
        if x.rank <= 1:
            return x
        batch = x.shape[0]
        return ml.reshape(x, [batch, x.size / batch])


class Sequential(Module):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


class RNNCell(Module):

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = ml.Parameter(_rand_uniform([input_size, hidden_size], -limit, limit))
        self.weight_hh = ml.Parameter(_rand_uniform([hidden_size, hidden_size], -limit, limit))
        if bias:
            self.bias_ih = ml.Parameter(_rand_uniform([hidden_size], -limit, limit))
            self.bias_hh = ml.Parameter(_rand_uniform([hidden_size], -limit, limit))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, h=None):
        if h == None:
            h = _init_hidden_like_input(x, self.hidden_size)

        out = ml.add(ml.matmul(x, self.weight_ih), ml.matmul(h, self.weight_hh))
        if self.bias_ih != None:
            out = ml.add(out, self.bias_ih)
            out = ml.add(out, self.bias_hh)
        return ml.tanh(out)


class RNN(Module):

    def __init__(self, input_size, hidden_size, bias=True, batch_first=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = RNNCell(input_size, hidden_size, bias)

    def forward(self, x, h0=None):
        result = _run_recurrent_sequence(self.cell, x, h0, self.batch_first)
        return [result[0], _expand_hidden_for_return(result[1])]


class LSTMCell(Module):

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit = 1.0 / math.sqrt(hidden_size)
        gate_width = hidden_size * 4
        self.weight_ih = ml.Parameter(_rand_uniform([input_size, gate_width], -limit, limit))
        self.weight_hh = ml.Parameter(_rand_uniform([hidden_size, gate_width], -limit, limit))
        if bias:
            self.bias_ih = ml.Parameter(_rand_uniform([gate_width], -limit, limit))
            self.bias_hh = ml.Parameter(_rand_uniform([gate_width], -limit, limit))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def _split_gates(self, gates):
        values = gates.tolist()
        hidden = self.hidden_size

        if gates.rank == 1:
            return [
                ml.tensor(values[0:hidden]),
                ml.tensor(values[hidden:hidden * 2]),
                ml.tensor(values[hidden * 2:hidden * 3]),
                ml.tensor(values[hidden * 3:hidden * 4]),
            ]

        i_gate = []
        f_gate = []
        g_gate = []
        o_gate = []
        row = 0
        while row < len(values):
            current = values[row]
            i_gate.append(current[0:hidden])
            f_gate.append(current[hidden:hidden * 2])
            g_gate.append(current[hidden * 2:hidden * 3])
            o_gate.append(current[hidden * 3:hidden * 4])
            row = row + 1

        return [ml.tensor(i_gate), ml.tensor(f_gate), ml.tensor(g_gate), ml.tensor(o_gate)]

    def forward(self, x, state=None):
        if state == None:
            h = _init_hidden_like_input(x, self.hidden_size)
            c = _init_hidden_like_input(x, self.hidden_size)
        else:
            h = state[0]
            c = state[1]

        gates = ml.add(ml.matmul(x, self.weight_ih), ml.matmul(h, self.weight_hh))
        if self.bias_ih != None:
            gates = ml.add(gates, self.bias_ih)
            gates = ml.add(gates, self.bias_hh)

        i_gate, f_gate, g_gate, o_gate = self._split_gates(gates)
        input_gate = ml.sigmoid(i_gate)
        forget_gate = ml.sigmoid(f_gate)
        candidate = ml.tanh(g_gate)
        output_gate = ml.sigmoid(o_gate)

        c_next = ml.add(ml.mul(forget_gate, c), ml.mul(input_gate, candidate))
        h_next = ml.mul(output_gate, ml.tanh(c_next))
        return [h_next, c_next]


class LSTM(Module):

    def __init__(self, input_size, hidden_size, bias=True, batch_first=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = LSTMCell(input_size, hidden_size, bias)

    def forward(self, x, state=None):
        current_state = _as_state_pair(state)
        result = _run_recurrent_sequence(self.cell, x, current_state, self.batch_first)
        hn = _expand_hidden_for_return(result[1][0])
        cn = _expand_hidden_for_return(result[1][1])
        return [result[0], [hn, cn]]


class GRUCell(Module):

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit = 1.0 / math.sqrt(hidden_size)
        gate_width = hidden_size * 3
        self.weight_ih = ml.Parameter(_rand_uniform([input_size, gate_width], -limit, limit))
        self.weight_hh = ml.Parameter(_rand_uniform([hidden_size, gate_width], -limit, limit))
        if bias:
            self.bias_ih = ml.Parameter(_rand_uniform([gate_width], -limit, limit))
            self.bias_hh = ml.Parameter(_rand_uniform([gate_width], -limit, limit))
        else:
            self.bias_ih = None
            self.bias_hh = None

    def _split_gates(self, gates):
        values = gates.tolist()
        hidden = self.hidden_size

        if gates.rank == 1:
            return [
                ml.tensor(values[0:hidden]),
                ml.tensor(values[hidden:hidden * 2]),
                ml.tensor(values[hidden * 2:hidden * 3]),
            ]

        reset_gate = []
        update_gate = []
        new_gate = []
        row = 0
        while row < len(values):
            current = values[row]
            reset_gate.append(current[0:hidden])
            update_gate.append(current[hidden:hidden * 2])
            new_gate.append(current[hidden * 2:hidden * 3])
            row = row + 1

        return [ml.tensor(reset_gate), ml.tensor(update_gate), ml.tensor(new_gate)]

    def forward(self, x, h=None):
        if h == None:
            h = _init_hidden_like_input(x, self.hidden_size)

        gates = ml.add(ml.matmul(x, self.weight_ih), ml.matmul(h, self.weight_hh))
        if self.bias_ih != None:
            gates = ml.add(gates, self.bias_ih)
            gates = ml.add(gates, self.bias_hh)

        reset_gate, update_gate, new_gate = self._split_gates(gates)
        reset_gate = ml.sigmoid(reset_gate)
        update_gate = ml.sigmoid(update_gate)
        candidate = ml.tanh(ml.add(new_gate, ml.mul(reset_gate, h)))
        keep = ml.sub(1, update_gate)
        return ml.add(ml.mul(keep, candidate), ml.mul(update_gate, h))


class GRU(Module):

    def __init__(self, input_size, hidden_size, bias=True, batch_first=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = GRUCell(input_size, hidden_size, bias)

    def forward(self, x, h0=None):
        result = _run_recurrent_sequence(self.cell, x, h0, self.batch_first)
        return [result[0], _expand_hidden_for_return(result[1])]


class Adam:

    def __init__(self, params, lr=0.001, betas=None, eps=0.00000001):
        self.params = params
        self.lr = lr
        if betas == None:
            betas = (0.9, 0.999)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.step_count = 0
        self.m = []
        self.v = []

        for param in params:
            self.m.append(ml.zeros(param.shape, param.dtype, param.device))
            self.v.append(ml.zeros(param.shape, param.dtype, param.device))

    def zero_grad(self):
        ml.zero_grad(self.params)

    def step(self):
        self.step_count = self.step_count + 1
        ml.adam_step(self.params, self.m, self.v, self.lr, self.beta1, self.beta2, self.eps, self.step_count)

    def state_dict(self):
        return {
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "step_count": self.step_count,
            "m": _serialize_state_value(self.m),
            "v": _serialize_state_value(self.v),
        }

    def load_state_dict(self, state, map_location=None, strict=True):
        missing = []
        expected = ["lr", "beta1", "beta2", "eps", "step_count", "m", "v"]
        i = 0
        while i < len(expected):
            key = expected[i]
            if key not in state:
                missing.append(key)
            i = i + 1

        unexpected = []
        for key in _sorted_keys(state):
            if key not in expected:
                unexpected.append(key)

        if strict and len(missing) > 0:
            raise ValueError("Missing optimizer state keys: " + ", ".join(missing))
        if strict and len(unexpected) > 0:
            raise ValueError("Unexpected optimizer state keys: " + ", ".join(unexpected))

        if "lr" in state:
            self.lr = state["lr"]
        if "beta1" in state:
            self.beta1 = state["beta1"]
        if "beta2" in state:
            self.beta2 = state["beta2"]
        if "eps" in state:
            self.eps = state["eps"]
        if "step_count" in state:
            self.step_count = int(state["step_count"])
        if "m" in state:
            self.m = _deserialize_state_value(state["m"], self.m, map_location)
        if "v" in state:
            self.v = _deserialize_state_value(state["v"], self.v, map_location)

        if strict and len(self.m) != len(self.params):
            raise ValueError("Optimizer state does not match parameter count.")
        if strict and len(self.v) != len(self.params):
            raise ValueError("Optimizer state does not match parameter count.")

        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    def save(self, path):
        payload = {
            "format": "tensorpy.nn.adam.v1",
            "state": self.state_dict(),
        }
        io.write_text(path, json.dumps(payload))
        return path

    def load(self, path, map_location=None, strict=True):
        payload = json.loads(io.read_text(path))
        if type(payload) != "dict" or payload.get("format") != "tensorpy.nn.adam.v1":
            raise ValueError("Invalid TensorPy Adam checkpoint.")
        return self.load_state_dict(payload.get("state"), map_location, strict)


class LogisticRegression(Module):

    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, out_features)

    def forward(self, x):
        return self.linear.forward(x)


class MLP(Module):

    def __init__(self, in_features, hidden_features, out_features):
        self.layers = Sequential([
            Linear(in_features, hidden_features),
            ReLU(),
            Linear(hidden_features, out_features),
        ])

    def forward(self, x):
        return self.layers.forward(x)


class SimpleCNN(Module):

    def __init__(self, image_size=28, in_channels=1, conv_channels=4, kernel_size=3, num_classes=10):
        conv_size = image_size - kernel_size + 1
        flattened = conv_channels * conv_size * conv_size
        self.features = Sequential([
            Conv2d(in_channels, conv_channels, kernel_size),
            ReLU(),
            Flatten(),
        ])
        self.classifier = Linear(flattened, num_classes)

    def forward(self, x):
        x = self.features.forward(x)
        return self.classifier.forward(x)


def save(module, path):
    if not _is_module(module) and not isinstance(module, Adam):
        raise TypeError("save() expects an nn.Module or nn.Adam.")
    return module.save(path)


def load(module, path, map_location=None, strict=True):
    if not _is_module(module) and not isinstance(module, Adam):
        raise TypeError("load() expects an nn.Module or nn.Adam.")
    return module.load(path, map_location, strict)
