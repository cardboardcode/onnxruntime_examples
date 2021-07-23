cmake_minimum_required(VERSION 2.8)
project( yolov3_onnxruntime )

set(CMAKE_CXX_STANDARD_LIBRARIES -ljpeg)

find_package( OpenCV REQUIRED )
find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

set(onnxruntime_INCLUDE_DIRS
  /usr/local/include/onnxruntime
  /usr/local/include/onnxruntime/core/session
)

include_directories(include
 ${onnxruntime_INCLUDE_DIRS}
)

add_executable(yolov3 src/main.cpp src/image.cpp src/ascii.cpp src/box.cpp)
target_link_libraries(yolov3 ${onnxruntime_LIBS} ${OpenCV_LIBS})
