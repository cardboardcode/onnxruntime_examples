#ifndef BOX_H
#define BOX_H

#include<vector>

typedef struct{
    float x, y, w, h;
} Box;

// from: box.h
typedef struct detection {
    Box bbox;
    float conf;
    int c;
    float prob;
} detection;

typedef struct{
    float dx, dy, dw, dh;
} Dbox;



Box float_arry_to_box(float *f);
Box float_to_box(float fx, float fy, float fw, float fh);
float box_iou(Box a, Box b);
Dbox diou(Box a, Box b);

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(Box a, Box b);
float box_union(Box a, Box b);
void filter_boxes_nms(std::vector<detection> &det, int nBoxes, float th_nms);

// float box_rmse(Box a, Box b);
// void do_nms(Box *boxes, float **probs, int total, int classes, float thresh);
// void do_nms_sort_v2(Box *boxes, float **probs, int total, int classes, float thresh);
// void do_nms_sort(detection *dets, int total, int classes, float thresh); // v3
// Box decode_box(Box b, Box anchor);
// Box encode_box(Box b, Box anchor);

#endif
