#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

Box float_arr_to_box(float *f)
{
    Box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}

Box float_to_box(float fx, float fy, float fw, float fh)
{
    Box b;
    b.x = fx;
    b.y = fy;
    b.w = fw;
    b.h = fh;
    return b;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Box a, Box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(Box a, Box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}


float box_iou(Box a, Box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

/*****************************************
* Function Name	: filter_boxes_nms
* Description	: Apply nms to get rid of overlapped rectangles.
* Arguments	:	std::vector<detection> &det: detected rectangles
*             int nBoxes : number of detections stored in det
* 						float th_nms : threshold for nms
* Return value	:
******************************************/
void filter_boxes_nms(std::vector<detection> &det, int nBoxes, float th_nms){
	int count = nBoxes;
	int i, j;
	for (i=0;i<count;i++){
		Box a = det[i].bbox;
		for (j=0;j<count;j++){
			if (i == j) continue;
			if (det[i].c != det[j].c) continue;
			Box b = det[j].bbox;
			float b_intersection = box_intersection(a, b);
			if (box_iou(a, b) > th_nms ||
					b_intersection >= a.h*a.w-1 ||
					b_intersection >= b.h*b.w - 1 )
			{
				if (det[i].prob > det[j].prob) det[j].prob= 0;
				else det[i].prob= 0;
			}
		}
	}
}

/*
Dbox derivative(Box a, Box b)
{
    Dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}



float box_rmse(Box a, Box b)
{
    return sqrt(pow(a.x-b.x, 2) +
                pow(a.y-b.y, 2) +
                pow(a.w-b.w, 2) +
                pow(a.h-b.h, 2));
}

Dbox dintersect(Box a, Box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    Dbox dover = derivative(a, b);
    Dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

Dbox dunion(Box a, Box b)
{
    Dbox du;

    Dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    Box a = {0, 0, 1, 1};
    Box dxa= {0+.0001, 0, 1, 1};
    Box dya= {0, 0+.0001, 1, 1};
    Box dwa= {0, 0, 1+.0001, 1};
    Box dha= {0, 0, 1, 1+.0001};

    Box b = {.5, .5, .2, .2};
    Dbox di = dunion(a,b);
    //printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001F);
    yinter = (yinter - inter)/(.0001F);
    winter = (winter - inter)/(.0001F);
    hinter = (hinter - inter)/(.0001F);
    //printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    Box a = {0, 0, 1, 1};
    Box dxa= {0+.0001, 0, 1, 1};
    Box dya= {0, 0+.0001, 1, 1};
    Box dwa= {0, 0, 1+.0001, 1};
    Box dha= {0, 0, 1, 1+.0001};

    Box b = {.5, .5, .2, .2};
    Dbox di = dintersect(a,b);
    //printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001F);
    yinter = (yinter - inter)/(.0001F);
    winter = (winter - inter)/(.0001F);
    hinter = (hinter - inter)/(.0001F);
    //printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    Box a = {0, 0, 1, 1};
    Box dxa= {0+.00001, 0, 1, 1};
    Box dya= {0, 0+.00001, 1, 1};
    Box dwa= {0, 0, 1+.00001, 1};
    Box dha= {0, 0, 1, 1+.00001};

    Box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    //printf("%f\n", iou);
    Dbox d = diou(a, b);
    //printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001F);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001F);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001F);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001F);
    //printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

Dbox diou(Box a, Box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    Dbox di = dintersect(a,b);
    Dbox du = dunion(a,b);
    Dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_bbox;

int nms_comparator_v2(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort_v2(Box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = exCalloc( total * sizeof(sortable_bbox), PRI_2, &dummy_location );

    for(i = 0; i < total; ++i){
        s[i].index = i;
        s[i].class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator_v2);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            Box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                Box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    exFree(s);
}


int nms_comparator(const void *pa, const void *pb)
{
//    detection a = *(detection *)pa;
//    detection b = *(detection *)pb;
//    float diff = 0;
//    if (b.sort_class >= 0) {
//        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
//    }
//    else {
//        diff = a.objectness - b.objectness;
//    }
//    if (diff < 0) return 1;
//    else if (diff > 0) return -1;
    return 0;
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
//    for (i = 0; i <= k; ++i) {
//        if (dets[i].objectness == 0) {
//            detection swap = dets[i];
//            dets[i] = dets[k];
//            dets[k] = swap;
//            --k;
//            --i;
//        }
//    }
//    total = k + 1;
//
//    for (k = 0; k < classes; ++k) {
//        for (i = 0; i < total; ++i) {
//            dets[i].sort_class = k;
//        }
//        qsort(dets, total, sizeof(detection), nms_comparator);
//        for (i = 0; i < total; ++i) {
//            //printf("  k = %d, \t i = %d \n", k, i);
//            if (dets[i].prob[k] == 0) continue;
//            Box a = dets[i].bbox;
//            for (j = i + 1; j < total; ++j) {
//                Box b = dets[j].bbox;
//                if (box_iou(a, b) > thresh) {
//                    dets[j].prob[k] = 0;
//                }
//            }
//        }
//    }
}

void do_nms(Box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

Box encode_box(Box b, Box anchor)
{
    Box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

Box decode_box(Box b, Box anchor)
{
    Box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}*/
