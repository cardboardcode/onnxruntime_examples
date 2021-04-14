cmake_minimum_required(VERSION 2.8)
project( minimal_ort_cpp_example )

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)
set(onnxruntime_INCLUDE_DIRS
  /usr/local/include/onnxruntime
  /usr/local/include/onnxruntime/core/session
)

find_package(OpenCV REQUIRED)

set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/build/squeezenet.onnx")
if (NOT EXISTS "${FILE}")
  message(STATUS "[squeezenet.onnx] does not exist. Downloading now ...")
  execute_process(COMMAND wget -c https://github.com/onnx/models/raw/master/vision/classification/squeezenet/model/squeezenet1.0-8.onnx -O "${CMAKE_CURRENT_SOURCE_DIR}/build/squeezenet.onnx")
endif()

include_directories(include
 ${onnxruntime_INCLUDE_DIRS}
)

add_executable( yolov4_infer src/yolov4_infer.cpp )
target_link_libraries( yolov4_infer ${onnxruntime_LIBS} ${OpenCV_LIBS})
