import numpy as np
from PIL import Image, ImageDraw
import onnxruntime
import datetime
import subprocess
import cv2
import os

# Check if model is downloaded.
if not os.path.exists('yolov3-10.onnx'):
    subprocess.call("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx", shell=True)

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data



# Define the model and pass it the input image.
ort_session = onnxruntime.InferenceSession("yolov3-10.onnx")

# Enable GPU
# ort_session.set_providers(['CUDAExecutionProvider'])

a = datetime.datetime.now()

image = Image.open("input.jpg")

# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)

# Initialize class label.
classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

# Set preprocessed input data.
ort_inputs = {ort_session.get_inputs()[0].name: image_data, ort_session.get_inputs()[1].name: image_size}

# Run inference.
ort_outs = ort_session.run(None, ort_inputs)

boxes = ort_outs[0]
scores = ort_outs[1]
indices = ort_outs[2]

out_boxes, out_scores, out_classes = [], [], []
for idx_ in indices:
    out_classes.append(idx_[1])
    out_scores.append(scores[tuple(idx_)])
    idx_1 = (idx_[0], idx_[2])
    out_boxes.append(boxes[idx_1])

b = datetime.datetime.now()
delta = b - a

print ("[yolov3-10.onnx] took " , int(delta.total_seconds() * 1000), " milliseconds.")

# print ("out_classes = ", out_classes)
# print ("out_boxes = ", out_boxes)
# print ("out_scores = ", out_scores)

# Post-process
result = ImageDraw.Draw(image)

# print ("Objects detected: ")
for item_no in range(len(out_classes)):
    # Print out the detected object name
    item_name =  (classes[out_classes[item_no]])
    # Draw bounding box on the picture.
    x1, y1 = int(out_boxes[item_no][0]), int(out_boxes[item_no][1])
    x2, y2 = int(out_boxes[item_no][2]), int(out_boxes[item_no][3])
    shape = [(y1, x1), (y2,x2)]
    result.rectangle(shape, fill=None, outline ="red")
    # text = item_name
    result.text((y1,x1), item_name, fill=None, font=None, anchor=None, spacing=0, align="left")

# Show resulting inference output.
image.show()
