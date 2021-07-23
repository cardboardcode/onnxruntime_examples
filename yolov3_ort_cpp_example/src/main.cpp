#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <numeric>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
/*****************************************
* Includes
******************************************/
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <jpeglib.h>
#include <termios.h>
#include <math.h>
#include <iomanip>
#include <sstream>
//#include <CommandAllocatorRing.h>

#include "box.h"
#include "define.h"
#include "image.h"

using namespace cv;
using namespace std;
/*****************************************
* Global Variables
******************************************/
std::map<int,std::string> label_file_map;
char inference_mode = DETECTION;
int model=0;
char* save_filename = "../output.jpg";
const char* input_file = "../input.jpg";
const char* mat_out = "mat_out.jpg";


const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);


void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}


/*****************************************
* Function Name :  loadLabelFile
* Description       : Load txt file
* Arguments         :
* Return value  :
******************************************/
int loadLabelFile(std::string label_file_name)
{
    int counter = 0;
    std::ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        perror("error while opening file");
        return -1;
    }

    std::string line;
    while(std::getline(infile,line))
    {
        label_file_map[counter++] = line;
    }

    if (infile.bad())
    {
        perror("error while reading file");
        return -1;
    }

    return 0;
}

/*****************************************
* Function Name : timedifference_msec
* Description   :
* Arguments :
* Return value  :
******************************************/
static double timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

/*****************************************
* Function Name : print_box
* Description   : Function to printout details of single bounding box to standard output
* Arguments :   detection d: detected box details
*             int i : Result number
* Return value  :
******************************************/
void print_box(detection d, int i){
    printf("\x1b[4m"); //Change colour
    printf("\nResult %d\n", i);
    printf("\x1b[0m"); //Change the top first colour
    printf("\x1b[1m"); //Change the top first colour
    printf("Detected        : %s\n",label_file_map[d.c].c_str());//, detected
    printf("\x1b[0m"); //Change the colour to default
    printf("Bounding Box    : (X, Y, W, H) = (%.2f, %.2f, %.2f, %.2f)\n", d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h);
    //printf("Confidence (IoU): %.1f %%\n", d.conf*100); //not use in yolov3
    //printf("Probability     : %.1f %%\n",  d.prob*100); //not use in yolov3
    printf("Score           : %f %%\n", d.prob * d.conf * 100);
}

int main(int argc, char* argv[])
{
    //Config : inference mode
    inference_mode = DETECTION;
    //Config : model
    std::string model_name = "yolov3-10.onnx";
    std::string model_path= "../yolov3-10.onnx";

    printf("Start Loading Model %s\n", model_name.c_str());

    int img_sizex, img_sizey, img_channels;

    //Postprocessing Variables
    int count = 0;
    float th_conf = 0.5;
    float th_prob = 0.5;
    std::vector<detection> det;

    //Timing Variables
    struct timeval start_time, stop_time,start_time_post, stop_time_post;
    double diff, time_pre, time_post;

    //UNCOMMENT to use dog image as an input
    struct S_Pixel
    {
        unsigned char RGBA[3];
    };

    gettimeofday(&start_time, nullptr);

    stbi_uc * img_data = stbi_load(input_file, &img_sizex, &img_sizey, &img_channels, STBI_default);
    std::vector<float> input_tensor_values_shape(2);
    input_tensor_values_shape[0] = img_sizey;
    input_tensor_values_shape[1] = img_sizex;
    Image *imge = new Image(img_sizex, img_sizey, 3);

    const S_Pixel * imgPixels0(reinterpret_cast<const S_Pixel *>(img_data));

    for ( size_t c = 0; c < 3; c++){
        for ( size_t y = 0; y < img_sizey; y++){
            for ( size_t x = 0; x < img_sizex; x++){
                const int val(imgPixels0[y * img_sizex + x].RGBA[c]);
                imge->set((y*img_sizex+x)*3+c, val);
            }
        }
    }

    /////////////
    int sizex=416, sizey=416;
    int img_sizex_new,img_sizey_new;
    double scale;
    scale = ((double)sizex/(double)img_sizex) < ((double)sizey/(double)img_sizey) ? ((double)sizex/(double)img_sizex) : ((double)sizey/(double)img_sizey);
    img_sizex_new = (int)(scale * img_sizex);
    img_sizey_new = (int)(scale * img_sizey);
    //printf("img_sizex: %d\n",img_sizex);
    //printf("img_sizey: %d\n",img_sizey);
    //printf("scale: %f\n",scale);
    //printf("img_sizex_new: %d\n",img_sizex_new);
    //printf("img_sizey_new: %d\n",img_sizey_new);
    cv::Mat img = cv::imread(input_file, cv::IMREAD_COLOR);  // (720, 960, 3)
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(img_sizex_new,img_sizey_new));
    cv::Mat new_image(sizex,sizey, CV_8UC3, Scalar(128,128,128));

    img_resize.copyTo(new_image(cv::Rect((sizex - img_resize.cols)/2,(sizey - img_resize.rows)/2,img_resize.cols, img_resize.rows)));
    cv::imwrite(mat_out, new_image);
    //printf("cols: %d\n",new_image.cols);
    //printf("rows: %d\n",new_image.rows);
    stbi_uc * img_data_new = stbi_load(mat_out, &img_sizex, &img_sizey, &img_channels, STBI_default);
    //////////////


    const S_Pixel * imgPixels(reinterpret_cast<const S_Pixel *>(img_data_new));

    //Config: label txt
    std::string filename("../coco_classes.txt");

    //ONNX runtime: Necessary
    OrtEnv* env;
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

    //ONNX runtime: Necessary
    OrtSession* session;
    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetInterOpNumThreads(session_options, 2); //Multi-core
    CheckStatus(g_ort->CreateSession(env, model_path.c_str(), session_options, &session));

    size_t num_input_nodes;
    size_t num_output_nodes;
    OrtStatus* status;
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);

    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims_input;
    std::vector<int64_t> input_node_dims_shape;
    std::vector<int64_t> output_node_dims;
    //printf("\nCurrent Model is %s\n",model_name.c_str());
    //printf("Number of inputs = %zu\n", num_input_nodes);
    //printf("Number of outputs = %zu\n", num_output_nodes);


    for (size_t i = 0; i < num_output_nodes; i++) {
        ////    // print input node names
        char* output_name;
        status = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
        //printf("output %d : name=%s\n", i, output_name);
        output_node_names[i] = output_name;
     ////// print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        //printf("Output %d : type=%d\n", i, type);
        //// print input shapes/dims
        size_t num_dims = 3;
        //printf("Output %d : num_dims=%zu\n", i, num_dims);
        output_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);

        for (int j = 0; j < num_dims; j++) {
            //printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
        }

        g_ort->ReleaseTypeInfo(typeinfo);
    }
// Print Out Input details
    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++){
        // print input node names
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        //printf("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        //printf("Input %zu : type=%d\n", i, type);
        size_t num_dims;
        if(i == 0)
        {
            num_dims = 4;
            //printf("Input %zu : num_dims=%zu\n", i, num_dims);
            input_node_dims_input.resize(num_dims);
            g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims_input.data(), num_dims);
            if(input_node_dims_input[0]<0){//Necessary for  Tiny YOLO v3
                input_node_dims_input[0]=1;   //Change the first dimension from -1 to 1
            }
            if(input_node_dims_input[2]<0){//Necessary for  Tiny YOLO v3
                input_node_dims_input[2]=416;   //Change the first dimension from -1 to 416
            }
            if(input_node_dims_input[3]<0){//Necessary for  Tiny YOLO v3
                input_node_dims_input[3]=416;   //Change the first dimension from -1 to 416
            }
            //for (size_t j = 0; j < num_dims; j++) printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims_input[j]);
        }
        else if(i == 1)
        {
            num_dims = 2;
            //printf("Input %zu : num_dims=%zu\n", i, num_dims);
            input_node_dims_shape.resize(num_dims);
            g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims_shape.data(), num_dims);
            if(input_node_dims_shape[0]<0){//Necessary for  Tiny YOLO v3
                input_node_dims_shape[0]=1;   //Change the first dimension from -1 to 1
            }
            //for (size_t j = 0; j < num_dims; j++) printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims_shape[j]);
        }

        g_ort->ReleaseTypeInfo(typeinfo);
    }

    //ONNX: Prepare input container
    size_t input_tensor_size = img_sizex * img_sizey * 3;
    std::vector<float> input_tensor_values(input_tensor_size);
    std::vector<float> input_tensor_values_new(input_tensor_size);
    int frame_count = 0;
    size_t offs, c, y, x;
    std::map<float,int> result; //Output for classification

    //Transpose
    offs = 0;
    for ( c = 0; c < 3; c++){
        for ( y = 0; y < img_sizey; y++){
            for ( x = 0; x < img_sizex; x++, offs++){
                const int val(imgPixels[y * img_sizex + x].RGBA[c]);
                input_tensor_values[offs] = ((float)val)/255;
            }
        }
    }

    gettimeofday(&stop_time, nullptr);

    time_pre = timedifference_msec(start_time,stop_time);

    // create input tensor object from data values
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    std::vector<OrtValue* > input_tensor(input_node_names.size());

    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size*sizeof(float), input_node_dims_input.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[0]));
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values_shape.data(), 2*sizeof(float), input_node_dims_shape.data(), 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[1]));

    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor[0],&is_tensor));
    assert(is_tensor);
    CheckStatus(g_ort->IsTensor(input_tensor[1],&is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    // RUN: score model & input tensor, get back output tensor
    std::vector<OrtValue *> output_tensor(3);
    output_tensor[0] = NULL;
    output_tensor[1] = NULL;
    output_tensor[2] = NULL;

    gettimeofday(&start_time, nullptr);
    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), input_tensor.data(), 2, output_node_names.data(), 3, output_tensor.data()));
    gettimeofday(&stop_time, nullptr);

    CheckStatus(g_ort->IsTensor(output_tensor[0],&is_tensor));
    assert(is_tensor);
    CheckStatus(g_ort->IsTensor(output_tensor[1],&is_tensor));
    assert(is_tensor);
    CheckStatus(g_ort->IsTensor(output_tensor[2],&is_tensor));
    assert(is_tensor);

    diff = timedifference_msec(start_time,stop_time);

    gettimeofday(&start_time_post, nullptr);//start postproc timer

    // Get pointer to output tensor float values
    float* out1 = NULL;
    g_ort->GetTensorMutableData(output_tensor[0], (void**)&out1);
    float* out2 = NULL;
    g_ort->GetTensorMutableData(output_tensor[1], (void**)&out2);
    int* out3 = NULL;
    g_ort->GetTensorMutableData(output_tensor[2], (void**)&out3);


    if(loadLabelFile(filename) != 0)
    {
        fprintf(stderr,"Fail to open or process file %s\n",filename.c_str());
        delete imge;
        return -1;
    }

    int nb_class = label_file_map.size();

    if(out3 != NULL)
    {
        for(size_t i = 0; i<10000; i+=3){
            if((out3[i] < 0) || (out3[i] > 10647)) break; // for yolov3
            float ymin = out1[out3[i+2]*4];
            float xmin = out1[out3[i+2]*4+1];
            float ymax = out1[out3[i+2]*4+2];
            float xmax = out1[out3[i+2]*4+3];
            Box bb = float_to_box((xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin);
            float objectness = 1;
            int detected = out3[i+1];
            float score = out2[10647 * out3[i+1] + out3[i+2]];
            detection d = { bb, objectness , detected,score };
            det.push_back(d);
            count++;
        }
    }

    int i, j=0;
    //Render boxes on image and print their details
    for(i =0;i<count;i++){
        if(det[i].prob == 0) continue;
        j++;
        print_box(det[i], j);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << det[i].conf*det[i].prob;
        std::string result_str = label_file_map[det[i].c]+ " "+ stream.str();
        imge->drawRect((int)det[i].bbox.x, (int)det[i].bbox.y, (int)det[i].bbox.w, (int)det[i].bbox.h, (int)det[i].c, result_str.c_str());
    }
    gettimeofday(&stop_time_post, nullptr);//Stop postproc timer
    time_post = timedifference_msec(start_time_post,stop_time_post);
    printf("\n");
    printf("\x1b[36;1m");
    printf("Preprocessing Time: %.3f msec\n", time_pre);
    printf("Postprocessing Time: %.3f msec\n", time_post);


    //Save Image
    imge->save(save_filename);

    g_ort->ReleaseValue(input_tensor[0]);
    g_ort->ReleaseValue(input_tensor[1]);
    g_ort->ReleaseValue(output_tensor[0]);
    g_ort->ReleaseValue(output_tensor[1]);
    g_ort->ReleaseValue(output_tensor[2]);

    remove(mat_out);

    printf("\x1b[36;1m");
    printf("Prediction Time: %.3f msec\n\n", diff);
    printf("\x1b[0m");

    delete imge;
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("Written to [ output.jpg ].\n");
    printf("Done!\n");

    return 0;
}
