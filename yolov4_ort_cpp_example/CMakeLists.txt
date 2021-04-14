cmake_minimum_required(VERSION 2.8)
project( minimal_ort_cpp_example )

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)
set(onnxruntime_INCLUDE_DIRS
  /usr/local/include/onnxruntime
  /usr/local/include/onnxruntime/core/session
)

find_package(OpenCV REQUIRED)

set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/build/yolov4.onnx")
if (NOT EXISTS "${FILE}")
  message(STATUS "[yolov4.onnx] does not exist. Downloading now ...")
  execute_process(COMMAND wget -c https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx -O "${CMAKE_CURRENT_SOURCE_DIR}/build/yolov4.onnx")
endif()

include_directories(include
 ${onnxruntime_INCLUDE_DIRS}
)

add_executable( infer src/main.cpp )
target_link_libraries( infer ${onnxruntime_LIBS} ${OpenCV_LIBS})
