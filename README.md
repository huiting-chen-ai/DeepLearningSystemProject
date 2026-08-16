# Project: FFT-Conv2d
Implementing 2D convolution via the Fast Fourier Transform (FFT) — supporting both forward and backward passes.   
This is the final project. It is based on the code before:
- https://github.com/huiting-chen-ai/DeepLearningSystemHW0
- https://github.com/huiting-chen-ai/DeepLearningSystemHW1
- https://github.com/huiting-chen-ai/DeepLearningSystemHW2
- https://github.com/huiting-chen-ai/DeepLearningSystemHW3
- https://github.com/huiting-chen-ai/DeepLearningSystemHW4
- https://github.com/huiting-chen-ai/DeepLearningSystemHW4Extra

## Overview
Standard spatial convolution (nn.Conv2d) slides a kernel over the input, computing dot products at each position — O(H·W·kh·kw) per channel pair. Using the Convolution Theorem, this can be accelerated in the frequency domain: x ∗ k = IFFT( FFT(x) ⊙ FFT(k) ), which has complexity O(H·W·log(H·W)) 

## Current Status
Forward: complete  
Backward: in progress

### Forward Pass
***The forward pass implements FFTConv2d with the following pipeline***:  
1. Pad — Input and kernel are zero-padded to the same spatial size (H + kh - 1, W + kw - 1)  
2. FFT — Both are transformed to the frequency domain via 2D FFT  
3. Multiply — Element-wise complex multiplication, summed over input channels  
4. IFFT — Inverse transform back to spatial domain
Crop — Extract the valid convolution region  
5. Undilate — Apply stride subsampling (if stride > 1)

***Problem***:  
1. I use recursion in FFT and IFFT. It cuts the input
in half each time, so the input has to be the power of 2. I pad the input to fit the condition, but it would requires more storage (and potentially more time).
2. When I implementing FFT, I need to pass complex number, but my cc (and ndarray) doesn't support complex, so I modified cc to allow complex number and 
adjust python code to use complex number
3. I didn't fully understand how FFT and IFFT work at start. I truncate the result of FFT so it has the 
same shape as the input I pass to FFT, so IFFT can't get the correct result. The problem is solved easily after I realize that I should keep the result of FFT as it is.
4. I use torch.fft to check my result at start, but then I change to nn.Conv2d. I think nn.Conv2d promises the correct of the expected value.

***Verification***:
- **Forward pass correctness**: compared against `nn.Conv2d` (PyTorch) with random inputs; norm difference < 1e-3
- **FFT/IFFT correctness**: verified `ifft(fft(x)) ≈ x` within numerical precision

***Lesson learned***:
- FFT-based convolution is mathematically elegant but requires careful handling of padding and frequency-domain indexing.
- The FFT result must never be truncated; the full spectrum is needed for invertibility.
- Complex number support in a custom autograd framework requires non-trivial infrastructure changes.

### Backward Pass
***The backward pass is consist of backward pass of its components, which includes***:
1. gradient of FFT
2. gradient of IFFT
3. gradient of pad
4. gradient of multiply
5. gradient of undilate

***Problem***:  
The backward is still in progress, but it should be easier comparing to forward pass. Since the FFT is consist of several simple operations, I can get the gradient by implementing the gradient for the simple operations correctly. I would add a test for the gradient and modify the code later.