# Copyright 2018 Slightech Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.0)

project(ref)

get_filename_component(PRO_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

add_definitions(-DLOG_TAG=ref)

if(DEBUG)
  add_definitions(-DDEBUG)
  message(STATUS "Using macro DEBUG")
endif()

# flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(BOOST_LIBS -lboost_system -lboost_filesystem -lboost_regex )
set(OPENCV_LIBS -L /usr/local/lib -lopencv_calib3d -lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab)
set(LIBS -lcblas  -lhdf5_serial -lhdf5_serial_hl -lprotobuf  -lglog -lgflags )


#find_package(Caffe)
find_package(Boost REQUIRED COMPONENTS system filesystem regex thread) 
set(Caffe_INCLUDE_DIRS /disk1/model-caffe/caffe/include  /disk1/model-caffe/caffe/build/src/ )
set(Caffe_LIBRARIES  /disk1/model-caffe/caffe/build/lib/libcaffe.so )
set(Caffe_DEFINITIONS -DCPU_ONLY=1 )
find_package(OpenCV)
add_compile_options(-lpthread)
include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})
add_executable(ssd_detect ssd_detect.cpp detect.cpp)

target_link_libraries(ssd_detect  ${Caffe_LIBRARIES} ${OpenCV_LIBS}   ${Boost_LIBRARIES} -lglog -lgflags ) 

message(${Caffe_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBS})
message(${Caffe_LIBRARIES})
