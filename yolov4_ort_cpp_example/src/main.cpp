// For file processing
#include <iostream>
#include <fstream>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ONNX
#include <onnxruntime_cxx_api.h>

// // Reference Library Files
// #include "utils.h"

// Returns true if file exist.
bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

cv::Mat padding(cv::Mat &img, int width, int height)
{
    int m_iPaddingDimension;
    int w, h, x, y;
    float r_w = width / (img.cols * 1.0);
    float r_h = height / (img.rows * 1.0);
    if (r_h > r_w)
    {
        w = width;
        h = r_w * img.rows;
        x = 0;
        y = (height - h) / 2;

        std::cout << "Padding Rh" << std::endl;
    }
    else
    {
        w = r_h * img.cols;
        h = height;
        x = (width - w) / 2;
        y = 0;

        std::cout << "Padding Rw" << std::endl;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);

    if (r_h > r_w)
    {
        m_iPaddingDimension = re.size().height;
    }
    else
    {

        m_iPaddingDimension = re.size().width;
    }


    cv::Mat out(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

int main(int argc, char* argv[]) {


  std::string modelPath = "./yolov4.onnx";
  std::string inputImagePath = "./input.jpg";
  // Check if the file exists
  if (is_file_exist(modelPath.c_str())) {
    std::cout << "Yolov4 ONNX model - [FOUND]" << std::endl;
  } else {
    std::cout << "Yolov4 ONNX model - [MISSING]" << std::endl;
    return 1;
  }

  std::unique_ptr<Ort::Env> m_OrtEnv;
  Ort::SessionOptions m_OrtSessionOptions;
  std::unique_ptr<Ort::Session> m_OrtSession;
  std::vector<char *> m_inputNodeNames;
  std::vector<char *> m_outputNodeNames;

  m_OrtEnv = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "ort"));
  m_OrtSessionOptions.SetIntraOpNumThreads(1);
  m_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  m_OrtSession = std::make_unique<Ort::Session>(Ort::Session(*m_OrtEnv, modelPath.c_str(), m_OrtSessionOptions));

  uint8_t m_numInputs = m_OrtSession->GetInputCount();
  std::cout << "m_numInputs is " << unsigned(m_numInputs) << std::endl;

  m_inputNodeNames.reserve(m_numInputs);
  // m_inputTensorSizes.reserve(m_numInputs);

  uint8_t m_numOutputs = m_OrtSession->GetOutputCount();
  std::cout << "m_numOutputs is " << unsigned(m_numOutputs) << std::endl;

  m_outputNodeNames.reserve(m_numOutputs);
  // m_outputTensorSizes.reserve(m_numOutputs);


  std::vector<std::vector<int64_t>> m_inputShapes;
  std::vector<int64_t> m_inputTensorSizes;
  bool m_inputShapesProvided = false;
  Ort::AllocatorWithDefaultOptions m_ortAllocator;

  for (int i = 0; i < m_numInputs; i++) {
    // If m_inputShapes not initialized,
    // then look at m_session and derive.
    // Ensures that m_inputShapes is filled properly before use.
    if (!m_inputShapesProvided) {
      Ort::TypeInfo typeInfo = m_OrtSession->GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

      m_inputShapes.emplace_back(tensorInfo.GetShape());
    }

    const auto & curInputShape = m_inputShapes[i];

    m_inputTensorSizes.emplace_back(
      std::accumulate(
        std::begin(curInputShape),
        std::end(curInputShape),
        1,
        std::multiplies<int64_t>()));

    char * inputName = m_OrtSession->GetInputName(i, m_ortAllocator);
    m_inputNodeNames.emplace_back(strdup(inputName));
    m_ortAllocator.Free(inputName);
  }

  std::cout << "It has " << m_inputShapes.size() << " layer with " <<  m_inputShapes[0].size() << " dimensions." << std::endl;
  std::cout << "Input shapes: { ";
  for (int i = 0; i < m_inputShapes[0].size(); i++) {
    std::cout << unsigned(m_inputShapes[0][i]) << ", ";
  }
  std::cout << " }" << std::endl;

  cv::Mat input_image;
  cv::Mat input_image_2;

  if (is_file_exist(inputImagePath.c_str())) {
    input_image = cv::imread(inputImagePath.c_str());
    std::cout << "Input Image - [FOUND]" << std::endl;
  } else {
    std::cout << "Input Image - [MISSING]" << std::endl;
    return 1;
  }

  input_image.copyTo(input_image_2);

  // Preprocess input image into tensor-suitable format.
  int m_iMcols = input_image_2.cols;
  int m_iMrows = input_image_2.rows;

  std::vector<int64_t> m_viNumberOfBoundingBox;
  m_viNumberOfBoundingBox.clear();
  input_image_2 = padding(input_image_2, 416, 416);


  return 0;
}
