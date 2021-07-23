#ifndef IMAGE_H
#define IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <vector>
#include "define.h"
#include "ascii.h"
// #define JCS_RGB 3
#define clip8bit(v) ((unsigned char)((v)>255)?255:((v)<0)?0:(v))

// #include <>
class Image
{
  public:
    Image();
    Image(int w, int h, int c);
    ~Image();

      std::vector<unsigned char> img_buffer;

    int get_H();
    int get_W();
    int get_C();

    void set_H(int h);
    void set_W(int w);

    void resize(int new_w, int new_h);
    void yuv2rgb();
    void save(char* filename);
    void drawRect(int x, int y, int w, int h, int color, const char* str);
    unsigned char at(int a);
    void set(int a, unsigned char val);

  private:
    std::vector<unsigned char> apl_buffer;
    int img_h;
    int img_w;
    int img_c;
    void drawPoint(int x, int y, int color);
    void drawLine(int x0, int y0, int x1, int y1, int color);
    void WriteCharSmall(char code, int x, int y, int color, int backcolor);
    void WriteStringSmall(const char *pcode, int x, int y, int color, int backcolor);

};

#endif
