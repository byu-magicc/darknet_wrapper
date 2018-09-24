# Darknet Wrapper

## Introduction
This is a multithreaded library wrapper for Darknet's YOLO network. YOLO is a deep neural network (DNN)
for object detection and classification. For more information on YOLO, visit the [Darknet](https://pjreddie.com/darknet/yolo/) website.
This wrapper extends Darknet's capabilities by allowing it to be used in any c++ program. 
There are a few things the user needs to be aware of
* All images must be of the same size due to dynamic memory allocation during initial initialization. 
* The wrapper is multithreaded and can use as many cores as the computer has; however, it is suggested that the user only uses three threads. To explain why, you need
    to become familiar with the architecture. Basically there are three stages: retrieving and formating the new image, passing the image through the YOLO network, and 
    formatting and sending the image out. Currently the program only has one net used by all the threads and so the second stage is the bottle neck. The developer could have
    multiple nets, but each net takes up about 1G of memory and will utilize your entire GPU or CPU when being used. 

## Setup

The Darknet Wrapper builds as a library, and currently there is not installation method. It is suggested that you add it as a submodule into your current project and add
it as a subdirectory into your CMakeList.txt. You will also need to include its directories and link the library. An example is shown below.

``` CMakeList.txt

add_subdirectory(darknet_wrapper)


include_directories(
include
darknet_wrapper/include
darknet_wrapper/darknet/include
darknet_wrapper/darknet/src

)

target_link_libraries(target
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  darknet_wrapper
)

```

For a mull example you can look at the CMakeList.txt file in [darknet_ros](https://magiccvs.byu.edu/gitlab/darknet/darknet_ros/blob/master/CMakeLists.txt)

## API

The user only needs to be aware of a few function found in YoloObjectDetector

