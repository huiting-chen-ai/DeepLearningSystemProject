"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math
from needle.backend_ndarray import ndarray

class FFTConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.weight = Parameter(init.kaiming_uniform(in_channels, out_channels, 
                                                     (kernel_size, kernel_size, in_channels, out_channels),
                                                     device=device, dtype=dtype, requires_grad=True))
        if bias:
            bound = 1/np.sqrt(in_channels*kernel_size**2)
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound, device=device,
                                            dtype=dtype, requires_grad=True))
        else:
            self.bias = None

    def _next_pow2(self, x):
        return 1 << ((x - 1).bit_length())

    def forward(self, x: Tensor) -> Tensor:
        # x: N x C x H x W  (NCHW)
        N, C, H, W = x.shape
        assert C == self.in_channels

        # Compute output spatial size for 'same' padding (ceil for stride>1)
        out_H = math.ceil(H / self.stride)
        out_W = math.ceil(W / self.stride)

        conv_H = H + self.kernel_size - 1
        conv_W = W + self.kernel_size - 1

        in_padded = ndarray.full((N, C, conv_H, conv_W), 0.0, dtype="complex32", device=x.device)
        # in_padded[..., :H, :W] = x
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        in_padded[n, c, h, w] = x[n, c, h, w].astype("complex32")
        in_padded = ops.reshape(in_padded, (N*C, conv_H, conv_W))

        I_fft = ops.fft2d(in_padded)
        I_fft = ops.reshape(I_fft, (N, C, conv_H, conv_W))

        # Prepare kernel: shape (kh, kw, in_ch, out_ch) -> we need (in_ch, out_ch, fft_H, fft_W)
        kh = kw = self.kernel_size
        K = self.weight  # (kh, kw, in_ch, out_ch)
        # create zero-padded kernel in top-left and then FFT
        K_pad = ndarray.full((self.in_channels, self.out_channels, conv_H, conv_W), 0.0,
                                dtype="complex32", device=K.device)
        # place kernel at top-left; note flipping: convolution uses cross-correlation unless you flip kernel.
        # to implement convolution via multiplication we must place kernel so that linear convolution matches spatial conv.
        # easiest: flip kernel spatially before placing so that multiplication corresponds to conv.
        K = ops.transpose(K, (0, 2))
        K = ops.transpose(K, (1, 3))
        K_flipped = ops.flip(K, (2, 3))  # -> (in_ch, out_ch, kh, kw) flipped
        K_pad[..., :kh, :kw] = K_flipped.astype("complex32")

        # FFT of kernels across spatial dims (per in->out pair)
        K_pad = ops.reshape(K_pad, (self.in_channels*self.out_channels, conv_H, conv_W))
        K_fft = ops.fft2d(K_pad)   # shape (in_ch, out_ch, fft_H, fft_W)
        K_fft = ops.reshape(K_pad, (self.in_channels, self.out_channels, conv_H, conv_W))

        # Multiply in frequency domain and sum over in_channels:
        # For each sample n and out channel o:
        #   Y_fft[n, o] = sum_c I_fft[n, c] * K_fft[c, o]
        # We can do broadcasting/matrix multiply in freq domain.
        # Reshape to broadcast: I_fft -> (N, C, 1, Hf, Wf), K_fft -> (1, C, O, Hf, Wf)
        I_fft_b = ops.broadcast_to(ops.reshape(I_fft, (N, self.in_channels, 1, conv_H, conv_W)), (N, self.in_channels, self.out_channels, conv_H, conv_W))
        K_fft_b = ops.broadcast_to(ops.reshape(K_fft, (1, self.in_channels, self.out_channels, conv_H, conv_W)), (N, self.in_channels, self.out_channels, conv_H, conv_W))
        # elementwise multiply and sum over in_channels -> (N, out_ch, Hf, Wf)
        Y_fft = ops.sum(I_fft_b * K_fft_b, axis=1)  # sum over in_channels, result (N, out_ch, Hf, Wf)

        # inverse FFT to spatial domain

        y_padded = ops.ifft2d(ops.reshape(Y_fft, (N*self.out_channels, conv_H, conv_W)))  # complex result
        y_padded = ops.reshape(y_padded, (N, self.out_channels, conv_H, conv_W))

        # normalize: depending on your FFT conventions, if ifft applies 1/(Hf*Wf) then ok; otherwise divide here
        # Crop to convolution spatial size conv_H x conv_W (top-left)
        y_cropped = y_padded[..., :conv_H, :conv_W]   # shape (N, out_ch, conv_H, conv_W)


        # apply stride by subsampling
        if self.stride > 1:
            y_cropped = ops.undilate(y_cropped, (2, 3), self.stride-1)

        # add bias if present: bias shape (out_ch,) -> reshape to (1, out_ch, 1, 1)
        if self.bias is not None:
            b = ops.reshape(self.bias, (1, self.out_channels, 1, 1))
            b = ops.broadcast_to(b, y_same.shape)
            y_same = y_same + b

        # final output: ensure real dtype if inputs/weights are real (drop tiny imag part)
        out = ops.real(y_same)
        # return in NCHW: it already is (N, out_ch, H_out, W_out)
        return out

    