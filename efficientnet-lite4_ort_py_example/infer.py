import numpy as np
import math
import matplotlib.pyplot as plt
import onnxruntime as rt
import cv2
import json
import sys
import os
import subprocess

# Check if model is downloaded.
if not os.path.exists('efficientnet-lite4.onnx'):
    subprocess.call("wget https://github.com/onnx/models/raw/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4.onnx", shell=True)
# Check if labels_map.txt is downloaded.
if not os.path.exists('labels_map.txt'):
    subprocess.call("wget https://raw.githubusercontent.com/lukemelas/EfficientNet-PyTorch/master/examples/simple/labels_map.txt", shell=True)
# Check if test image is downloaded.
if not os.path.exists('img.jpg'):
    subprocess.call("wget https://github.com/lukemelas/EfficientNet-PyTorch/raw/master/examples/simple/img.jpg", shell=True)


# load the labels text file
labels = json.load(open("labels_map.txt", "r"))

# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

# read the image
fname = "img.jpg"
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# pre-process the image like mobilenet and resize it to 224x224
img = pre_process_edgetpu(img, (224, 224, 3))
plt.axis('off')
plt.imshow(img)
plt.show()

# create a batch of 1 (that batch size is buned into the saved_model)
img_batch = np.expand_dims(img, axis=0)

# load the model
sess = rt.InferenceSession("efficientnet-lite4.onnx")
# run inference and print results
results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
result = reversed(results[0].argsort()[-5:])
for r in result:
    print(r, labels[str(r-1)], results[0][r])
