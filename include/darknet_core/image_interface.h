/*
 * image_interface.h
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#ifndef IMAGE_INTERFACE_H
#define IMAGE_INTERFACE_H

#include "image.h"


/**
* \brief Loads images of the alphabet.
* \detail These images are used to draw the classification
*         of bounding boxes on the image.
* @see YoloObjectDetector::InitNetwork
* @see YoloObjectDetector::DrawDetections
*/

image **load_alphabet_with_file(char *datafile);

#endif
