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
its CMakeList.tx as a subdirectory into your CMakeList.txt. You will also need to include its directories and link the library. An example is shown below.

Clone the repo.
``` bash
$ git clone --recurse-submodules git@magiccvs.byu.edu:darknet/darknet_wrapper.git
``` 


Add to your project's CMakeList.txt
``` cmake

add_subdirectory(darknet_wrapper)


include_directories(
include
darknet_wrapper/include
darknet_wrapper/darknet/include
darknet_wrapper/darknet/src
...

)

target_link_libraries(target
  darknet_wrapper
  ...
)

```

For a full example you can look at the CMakeList.txt file in [darknet_ros](https://magiccvs.byu.edu/gitlab/darknet/darknet_ros/blob/master/CMakeLists.txt)

## Documentation

Darknet Wrapper is documented using Doxygen syntax. You can view the documentation
1. Cloning the repository.
2. Navigate to the root directory.
3. Run ``` doxygen Doxyfile ``` in the terminal. This will create an html directory containing the documentation.
4. Create a session using your favorite web browser. e.g., ```google-chrome html/index.html```

## API

The user only needs to be aware of a few function found in YoloObjectDetector:

``` c++

 /**
  * \detail Loads all of the labels, parameters, weights, sets up
  *         the YOLO network, and starts the ThreadScheduler.
  * @param labels_filename The absolute path to the labels file.
  * @param params_filename The absolute path to the parameters file.
  * @param config_filename The absolute path to the configure file.
  * @param weights_filename The absolute path to the weights file.
  */
  YoloObjectDetector(
    std::string labels_filename, 
    std::string params_filename,
    std::string config_filename,
    std::string weights_filename);

  /**
  * \detail Stops all of the threads running, and
  *         frees all of the dynamically allocated memory.
  */
  ~YoloObjectDetector();

  
  /**
  * \detail Users will call this function to pass the images
  *         they want to be processed to YOLO. The image will be cloned.
  *         All images passed in must be the same size. 
  * @param img The user's passed in image.
  * @param seq Unique identifier for the image. Usually is the image's sequence.
  */
  void ImageCallback(const cv::Mat& img, int seq);

  /**
  * \detail Users will pass in a callback function to this method. Once
  *         an image is done being processed, the users supplied callback
  *         function will be called and be passed an image with the bounding boxes
  *         drawn if common::DynamicParams::draw_detections is set, passed 
  *         an object of common::BoundingBoxes, and the image's sequence number.
  *         Note that currently passed image is a reference to the data and 
  *         will be overwritten in the in the future.
  * @param pt2func Pointer to the callback function
  * @see common::BoundingBoxes
  * @see common::DynamicParams 
  */
  template<class T>
  void Subscriber(void (T::*pt2func)(const cv::Mat&, const common::BoundingBoxes&, const int&), T* object);

  /**
  * \detail The dynamic parameters specified by common::DynamicParams
  *         can be changed during runtime. When you pass in the parameters
  *         ensure that all of them are set. This method is thread safe.
  * @see common::DynamicParams
  */
  void SetDynamicParams(const common::DynamicParams& params);
```

You can look at the Doxygen generated documentation to have a better view of it. To see an example 
of using this API, see the tests or the project [darknet_ros](https://magiccvs.byu.edu/gitlab/darknet/darknet_ros).

## Parameters

There are many types of parameters: network parameters, static parameters, and dynamic parameters. The network parameters
are used directly by the YOLO net and comprise label file, weights file, and cfg file. These files are stored in the sub directory
yolo\_network\_config. There are already some label and configuration files ready to use, but you will have to download the weights you want to use. 
You can download these weights from [Darknet](https://pjreddie.com/darknet/yolo/). 

The static and dynamic parameters need to be specified before runtime. An example file of these parameters is in params/default.yaml. This file is loaded using the library yaml-cpp. 
The difference between static and dynamic is the dynamic parameters can be changed during runtime using the method YoloObjectDetector::SetDynamicParams. This method is thread safe. When you 
call this function and pass in the dynamic parameters, you must initialize ever member variabl of common::DynamicParams or unexpected things might happen. I have tried to put in safety 
measures to guard against this, but just be cautious. 

## Testing

If you are a developer and make changes, you can test your changes using the created test files. There are 
curently two tests: a parameter test and a detection test. To build the test files set the cmake flag ```-DBUILD_TESTS=ON``` and run make. 
then run the test.
```
./test/parameters_test
./test/detection_test
```

The detection test will show the image containing the drawn images to be inspected.