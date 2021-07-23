#include "image.h"

Image::Image(){
  img_w = 0;
  img_h = 0;
  img_c = 3;
}

Image::Image(int w, int h, int c){
  img_buffer.reserve(w*h*c);
  apl_buffer.reserve(w*h*c);
  img_w = w;
  img_h = h;
  img_c = c;
}
Image::~Image(){
}


int Image::get_H(){
  return img_h;
}
int Image::get_W(){
  return img_w;
}
int Image::get_C(){
  return img_c;
}

void Image::set_H(int h){
  img_h = h;
}
void Image::set_W(int w){
  img_w = w;
}

void Image::resize(int new_w, int new_h){
  double scale_width = (double) new_w/(double)img_w;
  double scale_height = (double) new_h/(double)img_h;
  apl_buffer.clear();

  int pixel, nearest;
  int i, j;
  for (j=0;j<new_h;j++){
    for (i=0;i<new_w;i++){
      pixel = (j*new_w+i)*BIT_COUNT_RGB;
      nearest = (((int)(j/scale_height)*(img_w*BIT_COUNT_RGB) +((int)(i/scale_width)*BIT_COUNT_RGB)));
      apl_buffer.push_back(img_buffer[nearest]);
      apl_buffer.push_back(img_buffer[nearest+1]);
      apl_buffer.push_back(img_buffer[nearest+2]);
    }
  }
  set_H(new_h);
  set_W(new_w);
  img_buffer.resize(apl_buffer.size());
  std::copy(apl_buffer.begin(), apl_buffer.end(), img_buffer.begin());
}


/*****************************************
* Function Name	: yuv2rgb
* Description		: Function to convert YUV into RGB format
* Arguments			: none
* Return value	: none
******************************************/
void Image::yuv2rgb(){
    apl_buffer.clear();
    int xpos, ypos;
    signed long y=0, u=0, v=0;

    int offset = 0;
    int y_loc=0, uv_loc=1;
    for(ypos=0;ypos<img_h;ypos++){
		for(xpos=0;xpos<img_w;xpos++, offset++){
            y = (signed long) img_buffer[y_loc];
            y_loc+=2;
            if ((xpos&1) == 0){
                u = (signed long) img_buffer[uv_loc] - 128;
                uv_loc+=2;
                v = (signed long) img_buffer[uv_loc] - 128;
                uv_loc+=2;
            }
            apl_buffer.push_back(clip8bit(((y*256) + (v*358))/256));//R
            apl_buffer.push_back(clip8bit(((y*256) - (v*182) - (u*88))/256));//G
            apl_buffer.push_back(clip8bit(((y*256) 					+ (u*453))/256));//B
		}
	}
    img_buffer.resize(apl_buffer.size());
    std::copy(apl_buffer.begin(), apl_buffer.end(), img_buffer.begin());
}


/*****************************************
* Function Name	: save
* Description	  :
* Arguments	    : char* filename= name of output image file
* Return value	: none
******************************************/
void Image::save(char* filename){
  //Allocate JPEG Object and Error handler
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  //Set default value to error handler
  cinfo.err = jpeg_std_error(&jerr);
  //Initialize JPEG object
  jpeg_create_compress(&cinfo);
  //Set Output file
  FILE * fp_i = fopen(filename, "wb");
  if (fp_i == NULL){
    fprintf(stderr, "Cannot open %s\n", filename);
    exit(EXIT_FAILURE);
  }
  jpeg_stdio_dest(&cinfo, fp_i);

  //Set image parameter
  cinfo.image_width = img_w;
  cinfo.image_height = img_h;
  cinfo.input_components = img_c;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 75,TRUE);
  //Start compress
  jpeg_start_compress(&cinfo, TRUE);
  //Set RGB
  JSAMPARRAY img = (JSAMPARRAY) malloc (sizeof(JSAMPROW)*img_h);
  for (int j = 0;j<img_h;j++){
    img[j]=(JSAMPROW)malloc(sizeof(JSAMPLE)*3*img_w);
    for (int i = 0;i<img_w;i++){
      img[j][i*3+0]=img_buffer[(j*img_w+i)*3+0];
      img[j][i*3+1]=img_buffer[(j*img_w+i)*3+1];
      img[j][i*3+2]=img_buffer[(j*img_w+i)*3+2];
    }
  }
  //Write
  jpeg_write_scanlines(&cinfo, img, img_h);
  //Finish compress
  jpeg_finish_compress(&cinfo);
  //Discard JPEG object
  jpeg_destroy_compress(&cinfo);

  for(int i= 0;i<img_h;i++){
    free(img[i]);
  }
  free(img);
  fclose(fp_i);

}


void Image::drawPoint(int x, int y, int color){


	//Draw
	img_buffer[(y*img_w+x)*3]	 = (color>>16)& 0x000000FF;
	img_buffer[(y*img_w+x)*3+1] = (color>>8)	& 0x000000FF;
	img_buffer[(y*img_w+x)*3+2] = color			& 0x000000FF;
}


void Image::drawLine(int x0, int y0, int x1, int y1, int color){
	int dx = x1-x0;
	int dy = y1-y0;
	int sx = 1;
	int sy = 1;
	float de;
	int i;
	if (dx<0) {
		dx*=(-1);
		sx*=(-1);
	}

	if (dy<0) {
		dy*=(-1);
		sy*=(-1);
	}

	drawPoint(x0, y0, color);

	if (dx>dy){
		for (i=dx, de=i/2; i;i--){
			x0 += sx;
			de += dy;
			if(de>dx){
				de -= dx;
				y0 += sy;
			}
			drawPoint(x0, y0, color);
		}
	}
	else{
		for (i=dy, de=i/2; i;i--){
			y0 += sy;
			de += dx;
			if(de>dy){
				de -= dy;
				x0 += sx;
			}
			drawPoint(x0, y0, color);
		}
	}
}

void Image::drawRect(int x, int y, int w, int h, int color, const char* str){
	int color_data;
	switch (color){
    case RED:
      color_data = RED_DATA;
    break;
		case BROSSOM:
			color_data = BROSSOM_DATA;
		break;
		case MAGENTA:
			color_data = MAGENTA_DATA;
		break;
		case ORANGE:
			color_data = ORANGE_DATA;
		break;
		case PEACH:
			color_data = PEACH_DATA;
		break;
		case DARKPURPLE:
			color_data = DARKPURPLE_DATA;
		break;
		case WINE:
			color_data = WINE_DATA;
		break;
		case PURPLE:
			color_data = PURPLE_DATA;
		break;
		case VIOLET:
			color_data = VIOLET_DATA;
		break;
		case LILAC:
			color_data = LILAC_DATA;
		break;
		case MUSTARD:
			color_data = MUSTARD_DATA;
		break;
		case DEEPBLUE:
			color_data = DEEPBLUE_DATA;
		break;
		case LEAF:
			color_data = LEAF_DATA;
		break;
		case LIGHTBLUE:
			color_data = LIGHTBLUE_DATA;
		break;
		case NAVY:
			color_data = NAVY_DATA;
		break;
		case BLUE:
			color_data = BLUE_DATA;
		break;
		case DARKGREEN:
			color_data = DARKGREEN_DATA;
		break;
		case TEAL:
			color_data = TEAL_DATA;
		break;
		case SKYBLUE:
			color_data = SKYBLUE_DATA;
		break;
		case GREEN:
			color_data = GREEN_DATA;
		break;
	}

  int x_min = x-w/2;
  int y_min = y-h/2;
  int x_max = x+w/2-1;
  int y_max = y+h/2-1;

  //Make sure the bounding box is in the image range.
	if (x_min<0)       {x_min = 0;}
  if (img_w<= x_max) {x_max = img_w-1;}
  if (y_min<0)       {y_min = 0;}
  if (img_h<=y_max)  {y_max = img_h-1; }

	drawLine(x_min,    y_min, 	 x_max, 	y_min, 	color_data);
	drawLine(x_max,    y_min, 	 x_max, 	y_max, 	color_data);
	drawLine(x_max,	   y_max,	   x_min, 	y_max, 	color_data);
	drawLine(x_min,    y_max,	   x_min, 	y_min, 	color_data);
}


unsigned char Image::at(int a){
  return img_buffer[a];
}

void Image::set(int a, unsigned char val){
  img_buffer[a] = val;
}
