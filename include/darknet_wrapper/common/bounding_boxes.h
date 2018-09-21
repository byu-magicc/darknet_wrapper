#pragma once

#include <string>
#include <vector>
#include <chrono>

namespace darknet_wrapper { namespace common {

/** \struct Prediction
* \brief Contains the probability for several possible classifications
* for each detected object. 
* \detail Each detected object (BoundingBox) will
* have multiple predictions associated with it. Currently darknet_wrapper
* returns all classifications for a single objects that is above the 
* threshold. 
*
*/

struct Prediction {

	std::string class_label;        /**< The detected object's classification label. */
	float probability;              /**< The probability that the object is its classification label. */

};


/** \struct BoundingBox
* \brief Contains information for a bounding box or detected object.
* \detal This container holds information about predictions, bounding box's
* center position, bounding box's top left corner and bottom right corner 
* positions. These coordinates are in pixels. 
*@see Prediction
*/

struct BoundingBox {

	std::vector<Prediction> predictions;

	float xcent; /**< x pixel coordinate of the bounding box's center. */
	float ycent; /**< y pixel coordinate of the bounding box's center. */
	float xmin;  /**< x pixel coordinate of the bounding box's top left corner. */
	float ymin;  /**< y pixel coordinate of the bounding box's top left corner. */
	float xmax;  /**< x pixel coordinate of the bounding box's bottom right corner. */
	float ymax;  /**< y pixel coordinate of the bounding box's bottom right corner. */


};


/** \struct BoundingBoxes
* \brief Contains all of the bounding boxes in a single image. 
* \detail Container for all of the detected objects and image data
* with processing utilization. Keep in mind that darkent_core is 
* multithreaded. Images processed per second is not the same as 
* 1/process_time.
*@see BoundingBox
*/
struct BoundingBoxes {

	std::chrono::time_point<std::chrono::system_clock> start_time;      /**< Time when image started to be processed. */
	std::chrono::time_point<std::chrono::system_clock> stop_time;       /**< Time when image stopped being processed. */

	std::vector<BoundingBox> boxes;

	int img_w;           /**< Image width in pixels. */
	int img_h;           /**< Image height in pixels. */
	
	int seq;             /**< Image sequence number. */

	int num_boxes;       /**< Number of bounding boxes found in the image. */
	float process_time;  /**< Amount of time it took for the image to be processed. */
	float fps;           /**< Average number of images processed by Darkent per second. */
};

}}