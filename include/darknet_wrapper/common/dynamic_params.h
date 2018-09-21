#pragma once

#include <mutex>
#include <boost/thread.hpp>


namespace darknet_wrapper { namespace common {

/** \struct DynamicParams
* \brief Contains all of the parameters that can be changed dynamically during runtime.
* \detail This container is seperate from TDynamicParams so that other methods can use
* it to set the dynamic parameters via YoloObjectDetector::SetDynamicParams. BE SURE TO
* INITIIALIZE EVERY MEMBER VARIABLE WHEN CHANGING DYNAMIC PARAMS!!!
*/

struct DynamicParams {

  /**
  * \detail The constructor is used to check if certain values have been changed.
  * only changed values will be set. 
  */
  DynamicParams() : threshold(-999.0f), draw_detections(false), frame_stride(-999){}

  float threshold;      /**< Detection threshold for YOLO. The prediction of a 
                            labeled bounding box must be higher than this parameter to
                            be considered sufficiently accurate. @see BoundingBoxes*/
  bool draw_detections; /**< Indicates if the detected bounding boxes should be drawn. @see BoundingBoxes*/

  int frame_stride;     /**< Every frame_stide frames are processed. If set to one, every frame will be
                             processed. If set to two, every other frame will be processed. Etc. */

};


/** \class TDynamicParams
* \brief Stands for Thread Dynamic Parameters.
* \detail Dynamic parameters can be changed during run time by different
* threads. This is is a helper class to make an object of DynamicParams
* thread safe.
*/

class TDynamicParams {

public:

  TDynamicParams() = default;
  ~TDynamicParams() = default;

  /**
  * \brief Thread safe method of setting the threshold value.
  * @param threshold The new threshold value.
  * @see DynamicParams::threashold
  */
  void SetThreshold(float threshold);

  /**
  * \brief Thread safe method of retrieving the threshold value.
  * @see DynamicParams::threashold
  */
  float GetThreshold();

  /**
  * \brief Thread safe method of setting the draw detection flag.
  * @param draw_detections The new flag value.
  * @see DynamicParams::draw_detections
  */
  void SetDrawDetections(bool draw_detections);

  /**
  * \brief Thread safe method of retrieving the draw detection flag.
  * @see DynamicParams::draw_detections
  */
  bool GetDrawDetections();

  /**
  * \brief Thread safe method of setting the frame stride.
  * @param frame_stride The new frame stride.
  */
  void SetFrameStride(int frame_stride);

  /**
  * \brief Thread safe method of retrieving the frame stride.
  */
  int GetFrameStride();

private:

  boost::shared_mutex threshold_mutex_;
  boost::shared_mutex draw_mutex_;
  boost::shared_mutex frame_mutex_;

  DynamicParams d_params_; /**< Contains all of the dynamic parameters */

};

}}