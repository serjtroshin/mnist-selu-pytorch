# HW

- train & convert the model to onnx:
`bash create_onnx_model.sh`

Expected output:
```
...
Train Epoch: 1 [56960/60000 (95%)]      Loss: 0.328974
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.109459
Train Epoch: 1 [58240/60000 (97%)]      Loss: 0.262587
Train Epoch: 1 [58880/60000 (98%)]      Loss: 0.196293
Train Epoch: 1 [59520/60000 (99%)]      Loss: 0.274753

Test set: Average loss: 0.0729, Accuracy: 9798/10000 (98%)

graph torch-jit-export (
  %input.1[FLOAT, 1x1x28x28]
) initializers (
  %conv1.weight[FLOAT, 32x1x3x3]
  %conv1.bias[FLOAT, 32]
  %conv2.weight[FLOAT, 64x32x3x3]
  %conv2.bias[FLOAT, 64]
  %fc1.weight[FLOAT, 128x9216]
  %fc1.bias[FLOAT, 128]
  %fc2.weight[FLOAT, 10x128]
  %fc2.bias[FLOAT, 10]
) {
  %9 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%input.1, %conv1.weight, %conv1.bias)
  %10 = mySelu(%9)
  %11 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%10, %conv2.weight, %conv2.bias)
  %12 = mySelu(%11)
  %13 = MaxPool[kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]](%12)
  %14 = Flatten[axis = 1](%13)
  %15 = Gemm[alpha = 1, beta = 1, transB = 1](%14, %fc1.weight, %fc1.bias)
  %16 = mySelu(%15)
  %17 = Gemm[alpha = 1, beta = 1, transB = 1](%16, %fc2.weight, %fc2.bias)
  %18 = LogSoftmax[axis = 1](%17)
  return %18
}
```


