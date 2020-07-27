## What Is This?
This example to build the `CXX_Api_Sample.cpp` that can be found on the official [onnxruntime](https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp) repository in the most minimalistic manner possible.

## Setup
Follow the instructions below to start building the example.
```bash
cd minimal_ort_cpp_example
mkdir build && cd build
cmake ..
make
./CXX_Api_Sample
```

## Verify
It is is working if this is the output it produces.
```txt
WARNING: Since openmp is enabled in this build, this API cannot be used to configure intra op num threads. Please use the openmp environment variables to control the number of threads.
Using Onnxruntime C++ API
Number of inputs = 1
Input 0 : name=data_0
Input 0 : type=1
Input 0 : num_dims=4
Input 0 : dim 0=1
Input 0 : dim 1=3
Input 0 : dim 2=224
Input 0 : dim 3=224
Score for class [0] =  0.000045
Score for class [1] =  0.003846
Score for class [2] =  0.000125
Score for class [3] =  0.001180
Score for class [4] =  0.001317
Done!
```
