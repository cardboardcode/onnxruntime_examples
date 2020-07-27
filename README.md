## What Is This?
A repository contains a bunch of examples of getting onnxruntime up and running in C++ and Python.

There is a `README.md` under each example. So read that to get started on that example you want.

## Getting Started with [**onnxruntime**]

### Build for C++
You can't run any of the C++ examples here if you don't build the library first.

Follow the commands below to build it. Don't worry. Here's a **bash script** to make it smoother.
```bash
git clone https://github.com/cardboardcode/onnxruntime_examples.git
cd onnxruntme_examples
sudo chmod u+x install_onnx_runtime_cpu.sh
./install_onnx_runtime_cpu.sh
```

### Build for Python
Building the onnxruntime python library will be one of the most painful and complex process you will ever face in your life.

Nah. Just kidding. It's actually easier. Just follow the instructions below.
```bash
pip install --user onnxruntime
```
`Caution`: This is only for using CPU.

For more details, please refer to **[the official onnxruntime installation page](https://microsoft.github.io/onnxruntime/)** here.

## Acknowledgement
1. The installation script, `install_onnx_runtime_cpu.sh`, was written by **[xmba15](https://github.com/xmba15/onnx_runtime_cpp)** here. Check his/her repository out. Beware the lack of documentation though.
