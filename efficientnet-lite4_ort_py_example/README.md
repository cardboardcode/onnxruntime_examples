## What Is This?
This example builds off the instructions provided in the **onnxruntime** Model Zoo for `efficientnet-lite4`. The model can be verified to be valid and its prediction accuracy is adequate.

## Setup
Follow the instructions below to build its python dependencies.

This runs in `python3`.
```bash
pip install onnxruntime
pip install matplotlib
pip install opencv-python
```

## Verify

1. Run `infer.py`.
```bash
python infer.py
```
2. You should see the following result.
```txt
388 lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens 0.99857914
296 American black bear, black bear, Ursus americanus, Euarctos americanus 0.0004527254
294 cheetah, chetah, Acinonyx jubatus 0.00033851538
295 brown bear, bruin, Ursus arctos 0.00018831895
387 African elephant, Loxodonta africana 3.0347202e-05
```
