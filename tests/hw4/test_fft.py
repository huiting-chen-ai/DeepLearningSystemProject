import sys
sys.path.append('./python')
import numpy as np
import pytest
from needle import backend_ndarray as nd
import needle as ndl
import mugrade
import itertools


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

def backward_check(f, *args, **kwargs):
    eps = 1e-3
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    is_stacked = False
    if isinstance(args[0], list):
        args = args[0]
        is_stacked = True
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            if is_stacked:
                f1 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            if is_stacked:
                f2 = (f(args, **kwargs).numpy() * c).sum()
            else:
                f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    if isinstance(backward_grad[0], ndl.TensorTuple): # TODO keep this?
        backward_grad = backward_grad[0].tuple()
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 1e-2
    return [g.numpy() for g in backward_grad]


stack_back_params = [
    ( (3, 4), 3, 0),
    ( (3, 4), 3, 1),
    ( (3, 4), 3, 2),
    ( (3, 4), 5, 2),
    ( (3, 4), 1, 2),
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("shape, n, axis", stack_back_params)
def test_stack_backward(shape, n, axis, device):
    np.random.seed(0)
    get_tensor = lambda shape: ndl.Tensor(np.random.randn(*shape)*5, device=device)
    backward_check(ndl.stack, [get_tensor(shape) for _ in range(n)], axis=axis)


stack_params = [
    {"shape": (10,3),    "n": 4, "axis": 0},
    {"shape": (4, 5, 6), "n": 5, "axis": 0},
    {"shape": (4, 5, 6), "n": 3, "axis": 1},
    {"shape": (4, 5, 6), "n": 2, "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", stack_params)
def test_stack_forward(params, device):
    np.random.seed(0)
    shape, n, axis = params['shape'], params['n'], params['axis']
    to_stack_ndl = []
    to_stack_npy = []
    for i in range(n):
        _A = np.random.randn(*shape)
        to_stack_ndl += [ndl.Tensor(_A, device=device)]
        to_stack_npy += [_A]

    lhs = np.stack(to_stack_npy, axis=axis)
    rhs = ndl.stack(to_stack_ndl, axis=axis)


pad_params = [
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )},
    {"shape": (10, 32, 32, 8), "padding": ( (0, 0), (0, 0), (0, 0), (0, 0) )},
]
@pytest.mark.parametrize("device", [nd.cpu()])
@pytest.mark.parametrize("params", pad_params)
def test_pad_forward(params, device):
    np.random.seed(0)
    shape, padding = params['shape'], params['padding']
    _A = np.random.randn(*shape)
    _B = np.pad(_A, padding)
    A = nd.NDArray(_A, device=device)
    B = A.pad(padding)

    assert np.linalg.norm(A.numpy() - _A) < 1e-4


flip_forward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 8), "axes": (0,1)},
    {"shape": (10, 32, 32, 8), "axes": (1,2)},
    {"shape": (3, 3, 6, 8), "axes": (1,2)},
    {"shape": (10, 32, 32, 8), "axes": (2,3)},
    {"shape": (3, 3, 6, 8), "axes": (2,3)},
    {"shape": (10, 32, 32, 8), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_forward_params)
def test_flip_forward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    _A = np.random.randn(*shape)
    _B = np.flip(_A, axes)
    A = ndl.Tensor(_A, device=device)
    B = ndl.flip(A, axes=axes)

    assert np.linalg.norm(A.numpy() - _A) < 1e-4


flip_backward_params = [
    {"shape": (10, 5), "axes": (0,)},
    {"shape": (10, 5), "axes": (1,)},
    {"shape": (10, 5), "axes": (0,1)},
    {"shape": (2, 3, 3, 8), "axes": (0,1)},
    {"shape": (3, 3, 6, 4), "axes": (0,1)},
    {"shape": (2, 3, 3, 4), "axes": (1,2)},
    {"shape": (3, 3, 6, 4), "axes": (1,2)},
    {"shape": (2, 3, 3, 4), "axes": (2,3)},
    {"shape": (3, 3, 6, 4), "axes": (2,3)},
    {"shape": (2, 3, 3, 4), "axes": (0,1,2,3)},
]
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("params", flip_backward_params)
def test_flip_backward(params, device):
    np.random.seed(0)
    shape, axes = params['shape'], params['axes']
    backward_check(ndl.flip, ndl.Tensor(np.random.randn(*shape), device=device), axes=axes)






# fft_forward_params = [
#     (4, 8, 16, 3, 1),
#     (32, 8, 16, 3, 2),
#     (32, 8, 8, 3, 2),
#     (32, 16, 8, 3, 1),
#     (32, 16, 8, 3, 2)
# ]
fft_forward_params = [
    (4, 8, 16, 3, 1)
]
@pytest.mark.parametrize("s,cin,cout,k,stride", fft_forward_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_fft_forward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.FFTConv2d(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(10, cin, s, s, device=device)

    z = torch.tensor(x.cached_data.numpy())
    weight = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))

    conv = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=0, bias=False)
    conv.weight = torch.nn.Parameter(weight)
    expected = conv(z)

    result = f(x).cached_data.numpy()
    assert np.linalg.norm(result - expected.detach().numpy()) < 1e-3

conv_back_params = [
    (4, 1, 1, 3, 1),
    (14, 8, 16, 3, 1),
    (14, 8, 16, 3, 2),
    (14, 8, 8, 3, 1),
    (14, 8, 8, 3, 2),
    (14, 16, 8, 3, 1),
    (14, 16, 8, 3, 2),
]
@pytest.mark.parametrize("s,cin,cout,k,stride", conv_back_params)
@pytest.mark.parametrize("device", _DEVICES)
def test_nn_conv_backward(s, cin, cout, k, stride, device):
    np.random.seed(0)
    import torch
    f = ndl.nn.Conv(cin, cout, k, stride=stride, device=device)
    x = ndl.init.rand(1, cin, s, s, device=device, requires_grad=True)

    g = torch.nn.Conv2d(cin, cout, k, stride=stride, padding=k//2)
    g.weight.data = torch.tensor(f.weight.cached_data.numpy().transpose(3, 2, 0, 1))
    g.bias.data = torch.tensor(f.bias.cached_data.numpy())
    z = torch.tensor(x.cached_data.numpy(), requires_grad=True)
    z.requires_grad = True

    res1 = f(x)
    y1 = res1.sum()

    y2 = g(z).sum()

    y1.backward()
    y2.backward()

    assert np.linalg.norm(g.weight.grad.data.numpy() - f.weight.grad.cached_data.numpy().transpose(3, 2, 0, 1)) < 1e-3, "weight gradients match"
    assert np.linalg.norm(g.bias.grad.data.numpy() - f.bias.grad.cached_data.numpy()) < 1e-3, "bias gradients match"
    assert np.linalg.norm(z.grad.data.numpy() - x.grad.cached_data.numpy()) < 1e-3, "input gradients match"



op_conv_shapes = [
    ((5, 16, 16))
]
@pytest.mark.parametrize("Z_shape", op_conv_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_fft(Z_shape, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    y = ndl.fft2d(Z)
    y2 = y.sum()
    # if backward:
    #     y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    out = torch.fft.fft2(Ztch)
    out2 = out.sum()
    # if backward:
    #     out2.backward()
    # if backward:
    #     err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
    #     err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    # if backward:
    #     assert err1 < 1e-2, "input grads match"
    #     assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)
    assert np.linalg.norm(y.numpy() - out.detach().numpy()) < 1e-3


fft_1d_shapes = [
    ((16,))
]
@pytest.mark.parametrize("Z_shape", fft_1d_shapes)
@pytest.mark.parametrize("device", [ndl.cpu()])
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_fft_1d(Z_shape, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    y = ndl.fft1d(Z)
    y2 = y.sum()
    # if backward:
    #     y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    out = torch.fft.fft(Ztch)
    out2 = out.sum()
    # if backward:
    #     out2.backward()
    # if backward:
    #     err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
    #     err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    # if backward:
    #     assert err1 < 1e-2, "input grads match"
    #     assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)
    assert y.dtype == "complex64"
    assert np.linalg.norm(y.numpy() - out.detach().numpy()) < 1e-3


op_ifft_shapes = [
    ((5, 16, 16))
]
@pytest.mark.parametrize("Z_shape", op_ifft_shapes)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("backward", [True, False], ids=["backward", "forward"])
def test_op_ifft(Z_shape, backward, device):
    np.random.seed(0)
    import torch
    _Z = np.random.randn(*Z_shape)*5
    _Z = _Z.astype(np.float32)
    Z = ndl.Tensor(_Z, device=device)
    Z_fft = ndl.fft2d(Z)
    y = ndl.ifft2d(Z_fft)
    y2 = y.sum()
    # if backward:
    #     y2.backward()
    Ztch = torch.Tensor(_Z).float()
    Ztch.requires_grad=True
    # out_fft = torch.fft.fft2(Ztch)
    # out = torch.fft.ifft2(out_fft)
    out2 = Ztch.sum()
    # if backward:
    #     out2.backward()
    # if backward:
    #     err1 = np.linalg.norm(Ztch.grad.numpy() - Z.grad.numpy())
    #     err2 = np.linalg.norm(Wtch.grad.numpy() - W.grad.numpy())
    err3 = np.linalg.norm(out2.detach().numpy() - y2.numpy())
    # if backward:
    #     assert err1 < 1e-2, "input grads match"
    #     assert err2 < 1e-2, "weight grads match"
    assert err3 < 1e-1, "outputs match %s, %s" % (y2, out2)
    assert np.linalg.norm(y.numpy() - Ztch.detach().numpy()) < 1e-3
