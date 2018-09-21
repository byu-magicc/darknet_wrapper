#pragma once

// c++
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <stdlib.h>

// Boost
#include <boost/thread.hpp>
#include <boost/date_time.hpp>

// OpenCv
#include <opencv2/opencv.hpp>

// common
#include "darknet_core/common/params.h"
#include "darknet_core/common/dynamic_params.h"
#include "darknet_core/common/bounding_boxes.h"

// Darknet.
#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "darknet_core/image_interface.h"
#include <sys/time.h>
}

extern "C" void ipl_into_image(IplImage* src, image im);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void show_image_cv(image p, const char *name, IplImage *disp);

namespace yolo_core {



/**
* \struct ImgThread
* \brief Container for various image information.
* \detail There is one of these objects per thread 
*         to allow access to member variables without data
*         racing conflicts.
*/
struct ImgThread {

  // 
  // Touched by multiple threads
  //

  bool thread_available = true; /** < Flag that indicates if it is available for a thread. 
                                      If false, a thread is currently controlling it.*/
  bool has_img = false;         /** <Flag that indicates if the container has a valid image 
                                      for processing. */

  // Image information
  cv::Mat img;                  /** <The image given to YOLO to be processed. */

  //
  // Touched only by one thread
  //

  // Image information
  cv::Mat draw_img;             /** <Contains the image with the bounding
                                      boxes drawn on it. */

  common::BoundingBoxes bounding_boxes; /** < @see common::BoundingBoxes*/


}; 


/** 
* \class YoloObjectDetector
* \brief Finds objects on a given image using YOLO and returns the results.
* \detail This class uses Darknet YOLO to find objects in an image and
*         classify them. All images being processed must be the same size. 
*/

class YoloObjectDetector
{
public:

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
  void Subscriber(void (T::*pt2func)(cv::Mat&, common::BoundingBoxes&, int&), T* object);

  /**
  * \detail The dynamic parameters specified by common::DynamicParams
  *         can be changed during runtime. When you pass in the parameters
  *         ensure that all of them are set. This method is thread safe.
  * @see common::DynamicParams
  */
  void SetDynamicParams(const common::DynamicParams& params);

protected:

  /**
  * \detail Initializes all of the parameters in the parameter file
  *         and reads in the class labels.
  * @param labels_filename The absolute path of the labels file.
  * @param params_filename The absolute path of the parameter file.
  */
  bool InitParameters(std::string labels_filename, std::string params_filename);

  /**
  * \detail Allocates memory for img_threads_ and sets initial values
  *         to indicate that it is ready for a thread and doesn't have an image.
  */
  void InitThreadQueue();

  /**
  * \detail Creates the network according to the weights and configuration
  *         specified in the wieghts and configuration files. This method also
  *         loads images of the alphabet to draw classification labels onto the 
  *         image.
  */
  void InitNetwork();

  /**
  * \detail Darknet needs the images to be formated a certain way
  *         to pass through the network. This method allocates memory for 
  *         Darknet images. This is why all of the images need to be the same 
  *         size. Otherwise, we would have to allocate memory every time a new
  *         image is received. 
  * @param img The first image that is received. It is used to 
  *            get the dimensions of all other images and to 
  *            allocate memory for all Darknet images and Ipl images.
  */ 
  void InitFrames(const cv::Mat& img);



  /**
  * \brief Handles all of the thread scheduling.
  * \detail When a new image is received and a thread is available
  *         the ThreadScheduler will call Yolo on a new thread and pass 
  *         in the new image. This allows for multiple images to be processed
  *         simultaneously. The thread that runs this method is joined
  *         by the class' destructor.
  */ 
  void ThreadScheduler();

  /**
  * \detail This is the core method. It detects and classifies objects
  *         in the image, draws the detection, and sends the results to the user
  *         via Subscriber. 
  * @param img The user's supplied image.
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources.
  */
  void Yolo(const cv::Mat& img, int thread_index);

  /**
  * \detail Formats the image from type cv::Mat to darknet::image 
  *         through a convoluted process. 
  *         cv::Mat -> IplImage -> darknet::image -> darknet::image (resized) 
  *         The resized darknet image (darknet_letters_) is what is sent through YOLO.
  *         This method is called by YoloObjectDetector::Yolo
  * @param img The user's supplied image.
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  void FormatImage(const cv::Mat& img, int thread_index);

  /**
  * \detail Passes the darkent image (darknet_letters_) through the 
  *         YOLO net. For each detected object, there can be initially multiple
  *         redundant bounding boxes. e.g. the same object is classified by
  *         two different bounding box, that classify the object the same. These
  *         redundant boxes are removed. There still can be multiple boxes for
  *         the same object. For example, one bounding box can classify the object
  *         as a sheep and the other can classify the object as a horse. 
  *         Since there is only one YOLO net, this resource is shared by all of
  *         the threads so that only one thread can access it at a time. 
  *         This method is called by YoloObjectDetector::Yolo
  * @param img The user's supplied image.
  * @param dets Container for all of the detected objects in the image.
  *             This contains all of the bounding boxes.
  * @param nboxes The number of bounding boxes in the image. 
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  void DetectObjects(const cv::Mat& img, detection *&dets, int &nboxes, int thread_index);

  /**
  * \detail Draws all of the bounding boxes on the image. This new image
  *         is stored in ImgThread::draw_img. 
  *         This method is called by YoloObjectDetector::Yolo
  * @param dets Container for all of the detected objects in the image.
  *             This contains all of the bounding boxes.
  * @param nboxes The number of bounding boxes in the image. 
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  void DrawDetections(detection *dets, int nboxes, int thread_index);

  /**
  * \detail Formats all of the detected objects in a common::BoudningBoxes
  *         container to be passed to the user's callback function they supplied to
  *         Subscriber.
  *         This method is called by YoloObjectDetector::Yolo
  * @param img The user's supplied image.
  * @param dets Container for all of the detected objects in the image.
  *             This contains all of the bounding boxes.
  * @param nboxes The number of bounding boxes in the image. 
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources.
  */
  void FormBoundingBoxes(const cv::Mat& img, detection* dets, int& nboxes,int thread_index);

  /**
  * \detail Calls the user's callback function and passes them ImgThread::drawn_img,
  *         ImgThread::bounding_boxes, and ImgThread::seq. 
  *         This method is called by YoloObjectDetector::Yolo
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources.
  * @see common::BoundingBoxes
  * @see ImgThread
  */
  void SendData(int thread_index);

  /**
  * \brief Converts cv::Mat to cv IplImage
  */
  IplImage* CvMatToCvIplImage(const cv::Mat& img);

  /**
  * \detail Copies the data in the darknet image to cv IplImage.
  * It follows the method shown in darknet/image.c show_image_cv
  */
  void ImageToIpl(const image& darknet_img, IplImage *cv_img);

  /**
  * \detail gets the pixel value of a Darknet Image
  * @param image The darknet image from which to get a pixel
  * @param x The x-image pixel coordinate
  * @param y The y-image pixel coordinate
  * @param c The channgel of the image.
  */
  float GetPixel(image m, int x, int y, int c);

  /**
  * \detail Determines if an image should be processed according to the
  *         frame stride.
  *         This method is called by YoloObjectDetector::ImageCallback
  */
  bool ProcessImage();

  /**
  * \detail Returns the index of a thread that isn't
  *         currently processing an image or doesn't have
  *         an image to process.
  *         This method is called by YoloObjectDetector::ImageCallback
  */
  int GetAvailableThreadForImage();

  /**
  * \detail Returns the index of every thread that isn't
  *         currently processing an image but has
  *         an image to process.
  *         This method is called by YoloObjectDetector::ThreadScheduler
  */
  std::vector<int> GetAvailableThreadsForProcessing();

  /**
  * \detail Places a newly received image by YoloObjectDetector::ImageCallback
  *         on the image thread YoloObjectDetector::img_threads_. 
  *         This method is called by YoloObjectDetector::ImageCallback
  * @param img The user's supplied image.
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  * @param seq Unique identifier for the image. Usually is the image's sequence.
  */
  void SetImage(const cv::Mat& img, int thread_index, int seq);

  /**
  * \detail Indicates that the thread is done processing and image and
  *         is available to process another image. 
  *         This method is called by YoloObjectDetector::Yolo
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  void ReleaseThread(int thread_index);

  /**
  * \detail Sets a flag that indicate the current thread is busy.
  *         This method is called by YoloObjectDetector::ThreadScheduler
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  void OccupyThread(int thread_index);

  /**
  * \detail Returns a reference to the image that is thread safe.
  * @param thread_index The thread's unique identifier. It is used to 
  *        index multiple sources. 
  */
  cv::Mat& GetImage(int thread_index);

  /**
  * \brief Returns true is the main thread is stil running.
  * \detail Since the ThreadScheduler is running on a different thread
  * than the main program, ThreadScheduler needs to know when the program
  * is no longer supposed to be running so that it can stop all of the other
  * threads and then its own.
  * This method is called by YoloObjectDetector::ThreadScheduler
  */
  bool IsProgramRunning();

  /**
  * \brief Increments the number of images the program has processed.
  */
  void IncNumFramesProcessed();

  /**
  * \brief Returns the average number of images Darknet processes per second.
  */
  float GetFps();

  
  bool is_program_running_ = true;           /**< Indicates if the program should still be runnins. @see IsProgramRunning*/
  boost::shared_mutex mutex_node_running_;

  // Darknet Parameters
  network *net_ = NULL;                /**< The YOLO network. */
  int net_width_ = 0;                  /**< The width of the YOLO network. */
  int net_height_ = 0;                 /**< The height of the YOLO network. */
  char *weights_file_path_ = NULL;     /**< The absolute file path of the weights file. */
  char *config_file_path_ = NULL;      /**< The absolute file path of the config file.*/
  char *data_file_path_ = NULL;        /**< The absolute file path of the parameters file.*/
  char **class_labels_ = NULL;         /**< The absolute file path of the  class labels file.*/
  int num_classes_ = 0;                /**< The number of class labels. */
  image **alphabet_ = NULL;            /**< Images of all the alphabetical letters. @see DrawDetections*/
  float tree_threshold_ = .5;          /**< see darknet/region_layer.c line 364 */  
  float nms_ = 0.4;                    /**< Parameter used to remove similar bounding boxes */
  boost::mutex mutex_net_;

  // Scheduler
  std::thread scheduler_thread_;       /**< Thread for the thread scheduler. */

  // Computer info
  int CPUs_ = 0;                       /**< Number of CPUs being used for threading. */
  int num_threads_ = 0;                /**< Number of threads the user wants to use. 
                                            This number is saturated by the total number of CPUs. */

  // Parameters
  common::Params params_;              /**< Used to load the static and initial dynamic parameters from param yaml file. */
  common::Params cl_params_;           /**< Used to load the class labels from the labels yaml file. */
  common::TDynamicParams d_params_;    /**< Helper object to change dynamic parameters during runtime. */

  // Frames
  ImgThread* img_threads_ = NULL;          /**< Container for images and other data for each thread. */
  boost::shared_mutex mutex_img_threads_;  /**< */
  std::vector<image> darknet_imgs_;        /**< Darknet Image container for each thread. */
  std::vector<image> darknet_letters_;     /**< Resized Darknet Image container for each thread. */
  std::vector<IplImage *> ipls_;           /**< Container for IPL images. */
  cv::Size img_size_;                      /**< The size of the first image received. It is used to check that all
                                                sequential images are of the same size. */

  std::function<void(cv::Mat&, common::BoundingBoxes&, int&)> fn_ptr_; /**< Subscriber function pointer. 
                                                  It points to the passed in callback function.
                                                  @see Subscriber.*/
  boost::mutex send_mutex_;

  
  uint32_t num_frames_processed_ = 0; /**< Total number of images processed. */            
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  boost::shared_mutex mutex_fps_;

  bool frames_init_ = false; /**< Flag to indicate if darknet_imgs_, darknet_letters_, and ipl_ 
                                  have been allocated. */ 
  int frame_count_ = 0;      /**< Counts the number of frames skipped. */

};

template<class T>
void YoloObjectDetector::Subscriber(void (T::*pt2func)(cv::Mat&, common::BoundingBoxes&, int&), T* object) 
{
  std::cout << "subscribed!!" << std::endl;
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  fn_ptr_ = std::bind(pt2func,object, _1,_2,_3);

  if (!fn_ptr_)
  {
    std::cout << "YoloObjectDetector: WARNING: Function pointer contains nothing" << std::endl;
  }
}




} /* namespace yolo_core*/
