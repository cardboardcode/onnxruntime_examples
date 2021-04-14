## What Is This?
This example shows you how to run the onnx's yolov4 model with the python code instructions.

## Create Virtual Environment
```bash
conda create -n yolov4_onnx_py python=3.6 -y
conda activate yolov4_onnx_py
pip install onnxruntime opencv-python Pillow scipy
```

## Run
```bash
python infer.py
```

## Verify
You know it is working if you get the following output image after running `python infer.py` under **Run**.

![](./result.png)
