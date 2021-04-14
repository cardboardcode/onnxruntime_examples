#include <assert.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {


  std::string modelPath = "../yolov4.onnx";
  // Check if the file exists
  // Reference: https://stackoverflow.com/a/51536462
  if (__cplusplus == 201703L) std::cout << "C++17\n";
  else if (__cplusplus == 201402L) std::cout << "C++14\n";
  else if (__cplusplus == 201103L) std::cout << "C++11\n";
  else if (__cplusplus == 199711L) std::cout << "C++98\n";
  else std::cout << "pre-standard C++\n";

  // Yolov4 *yolov4;

  return 0;
}
