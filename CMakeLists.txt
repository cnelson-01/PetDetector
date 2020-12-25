cmake_minimum_required(VERSION 3.16)
project(PetDetector)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(PetDetector main.cpp yolo.cpp yolo.h)
target_link_libraries( PetDetector ${OpenCV_LIBS} )