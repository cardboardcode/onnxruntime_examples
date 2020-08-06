import colorsys
import cv2
import datetime
import numpy

from PIL import Image

import os
import onnxruntime as rt
import random
from scipy import special



# Check if model is downloaded.
if not os.path.exists('yolov4.onnx'):
    subprocess.call("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx", shell=True)

# this function is from tensorflow-yolov4-tflite/core/utils.py
def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = numpy.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes

def get_anchors(anchors_path, tiny=False):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = numpy.array(anchors.split(','), dtype=numpy.float32)
    return anchors.reshape(3, 3, 2)

def postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=[1,1,1]):
    '''define anchor boxes'''
    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = numpy.meshgrid(numpy.arange(output_size), numpy.arange(output_size))
        xy_grid = numpy.expand_dims(numpy.stack(xy_grid, axis=-1), axis=2)

        xy_grid = numpy.tile(numpy.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(numpy.float)

        pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
        pred_wh = (numpy.exp(conv_raw_dwdh) * ANCHORS[i])
        pred[:, :, :, :, 0:4] = numpy.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [numpy.reshape(x, (-1, numpy.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = numpy.concatenate(pred_bbox, axis=0)
    return pred_bbox


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, numpy.inf]
    pred_bbox = numpy.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = numpy.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = numpy.concatenate([numpy.maximum(pred_coor[:, :2], [0, 0]),
                                numpy.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = numpy.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = numpy.sqrt(numpy.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = numpy.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = numpy.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[numpy.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = numpy.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return numpy.concatenate([coors, scores[:, numpy.newaxis], classes[:, numpy.newaxis]], axis=-1)

def bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = numpy.array(boxes1)
    boxes2 = numpy.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = numpy.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = numpy.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = numpy.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = numpy.maximum(1.0 * inter_area / union_area, numpy.finfo(numpy.float32).eps)

    return ious

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = numpy.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = numpy.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[numpy.newaxis, :4], cls_bboxes[:, :4])
            weight = numpy.ones((len(iou),), dtype=numpy.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = numpy.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox(image, bboxes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = numpy.array(bbox[:4], dtype=numpy.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

# input
input_size = 416

original_image = cv2.imread("input.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = image_preprocess(numpy.copy(original_image), [input_size, input_size])
image_data = image_data[numpy.newaxis, ...].astype(numpy.float32)

import onnxruntime as rt
sess = rt.InferenceSession("yolov4.onnx")

output_name = sess.get_outputs()[0].name
input_name = sess.get_inputs()[0].name

detections = sess.run([output_name], {input_name: image_data})[0]
print("Output shape:", detections.shape)

ANCHORS = "./yolov4_anchors.txt"
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

ANCHORS = get_anchors(ANCHORS)
STRIDES = numpy.array(STRIDES)

a = datetime.datetime.now()

pred_bbox = postprocess_bbbox(numpy.expand_dims(detections, axis=0), ANCHORS, STRIDES, XYSCALE)
bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.50)
bboxes = nms(bboxes, 0.213, method='nms')
image = draw_bbox(original_image, bboxes)

b = datetime.datetime.now()
delta = b - a
print ("[yolov4.onnx] took " , int(delta.total_seconds() * 1000), " milliseconds.")

image = Image.fromarray(image)
image.show()
