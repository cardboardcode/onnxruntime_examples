#ifndef DEFINE_MACRO_H
#define DEFINE_MACRO_H

//define macro here


/*****************************************
* image.h
******************************************/
/*****************************************
* camera.h
******************************************/
#define CAMERA_WIDTH 640
#define CAMERA_HEIGHT 480
/*****************************************
* box.h
******************************************/

/*****************************************
* onnxruntime_inference_example
******************************************/
#define CLASSIFICATION 	0
#define DETECTION 	1
#define AUTO_CAP 	0
#define MANUAL_CAP 	1
#define DEBUG_MODE 	1

#define BIT_COUNT_RGB 3
#define BIT_COUNT_ARGB 4
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define clip8bit(v) ((unsigned char)((v)>255)?255:((v)<0)?0:(v))

#define RESNET50 0
#define YOLO 1
#define TINYYOLO 2
#define MOBILENET 4
#define YOLO_GRID_X 13
#define YOLO_GRID_Y 13
#define YOLO_NUM_BB 5



/*****************************************
* For Drawing
******************************************/
#define FONTDATA_WIDTH_SMALL  (6) //don't change
#define FONTDATA_HEIGHT_SMALL (8) //don't change

#define RED_DATA         0xFF0000u
#define BROSSOM_DATA     0xFF0066u
#define MAGENTA_DATA     0xFF00FFu
#define ORANGE_DATA      0xFF9900u
#define PEACH_DATA       0xFF6666u
#define DARKPURPLE_DATA  0xCC0066u
#define WINE_DATA        0x990000u
#define PURPLE_DATA      0x990099u
#define VIOLET_DATA      0x9900FFu
#define LILAC_DATA       0x9966CCu
#define MUSTARD_DATA     0x999900u
#define DEEPBLUE_DATA    0x330099u
#define LEAF_DATA        0x339900u
#define LIGHTBLUE_DATA   0x3399CCu
#define NAVY_DATA        0x000033u
#define BLUE_DATA        0x0000FFu
#define DARKGREEN_DATA   0x003300u
#define TEAL_DATA        0x006666u
#define SKYBLUE_DATA     0x0099FFu
#define GREEN_DATA       0x00FF00u
#define WHITE_DATA       0xFFFFFFu
#define BLACK_DATA       0x000000u

#define RED         6u  //car
#define BROSSOM     18u //train
#define MAGENTA     2u  //bird
#define ORANGE      3u  //boat
#define PEACH       4u  //bottle
#define DARKPURPLE  5u  //bus
#define WINE        19u //tvmonitor
#define PURPLE      7u  //cat
#define VIOLET      8u  //chair
#define LILAC       9u  //cow
#define MUSTARD     10u //diningtable
#define DEEPBLUE    11u //dog
#define LEAF        12u //horse
#define LIGHTBLUE   13u //motorbike
#define NAVY        14u //person
#define BLUE        15u //pottedplant
#define DARKGREEN   16u //sheep
#define TEAL        17u //sofa
#define SKYBLUE     0u  //aeroplane
#define GREEN       1u  //bicycle
#define WHITE       20u
#define BLACK       21u
#define TRANSPARENT 22u

// #define MUSTARD_DATA     0x999900u
// #define BLUE_DATA        0x0000FFu
// #define GREEN_DATA       0x00FF00u
// #define RED_DATA         0xFF0000u
// #define ORANGE_DATA      0xFF9900u
// #define CYAN_DATA        0x00FFFFu
// #define YELLOW_DATA      0xFFFF00u
// #define MAGENTA_DATA     0xFF00FFu
// #define DEEPSKYBLUE_DATA 0x00BFFFu
// #define DARKGREEN_DATA   0x006400u
// #define PURPLE_DATA      0x800080u
// #define LIME_DATA        0x33FF00u
// #define GRAY_DATA        0x808080u
// #define BROWN_DATA       0xA52A2Au
// #define GOLD_DATA        0xFFD780u
// #define MAROON_DATA      0x800000u
// #define OLIVE_DATA       0x808000u
// #define TEAL_DATA        0x008080u
// #define NAVY_DATA        0x000080u
// #define LIGHTPURPLE_DATA 0xCC99FFu
// #define WHITE_DATA       0xFFFFFFu
// #define BLACK_DATA       0x000000u

// #define MUSTARD     0u
// #define BLUE        1u
// #define GREEN       2u
// #define RED         3u
// #define ORANGE      4u
// #define CYAN        5u
// #define YELLOW      6u
// #define MAGENTA     7u
// #define DEEPSKYBLUE 8u
// #define DARKGREEN   9u
// #define PURPLE      10u
// #define LIME        11u
// #define GRAY        12u
// #define BROWN       13u
// #define GOLD        14u
// #define MAROON      15u
// #define OLIVE       16u
// #define TEAL        17u
// #define NAVY        18u
// #define LIGHTPURPLE 19u
// #define WHITE       20u
// #define BLACK       21u
// #define TRANSPARENT 22u

#endif
