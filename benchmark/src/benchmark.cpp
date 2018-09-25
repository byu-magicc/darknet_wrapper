#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>


#include <opencv2/opencv.hpp>
#include <darknet_wrapper/common/dynamic_params.h>
#include <darknet_wrapper/common/bounding_boxes.h>
#include <darknet_wrapper/YoloObjectDetector.h>

#ifdef WRAPPER_FILE_PATH
const std::string kTestsDirPath = WRAPPER_FILE_PATH;

	std::string labels_filename = kTestsDirPath + "/yolo_network_config/labels/yolov3-tiny.yaml";
	std::string config_filename = kTestsDirPath + "/yolo_network_config/cfg/yolov3-tiny.cfg";
	std::string weights_filename = kTestsDirPath + "/yolo_network_config/weights/yolov3-tiny.weights";
	std::string params_filename = kTestsDirPath + "/benchmark/param/param.yaml";

#else
#error Path of WRAPPER_FILE_PATH repository is not defined in CMakeLists.txt.
#endif

#ifdef DARKNET_FILE_PATH
const std::string kDarknetFilePath_ = DARKNET_FILE_PATH;
std::string dog_file = kDarknetFilePath_ +"/data/dog.jpg";
std::string eagle_file = kDarknetFilePath_ +"/data/eagle.jpg";
std::string horses_file = kDarknetFilePath_ +"/data/horses.jpg";
std::string person_file = kDarknetFilePath_ +"/data/person.jpg";
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_wrapper {

class UtilBenchmark {

public:

	UtilBenchmark()
	{

		// Change only these two parameters
		run_time_ = 10;
		sleep_time_ = 100;

		////////////////////////////
		// Dont touch anything else
		////////////////////////////


		img_ = cv::imread(dog_file,CV_LOAD_IMAGE_COLOR);

		// Ensures the image exists
		if (!img_.data) {

			std::cout << "image not found" << std::endl;
			return;
		}

		yolo_.Initialize(
			labels_filename,
			params_filename,
			config_filename,
			weights_filename);

		// Create the YOLO subscriber
		yolo_.Subscriber(&UtilBenchmark::ImageCallback,this);

		// Change the parameters
		darknet_wrapper::common::DynamicParams d_params;
		d_params.threshold = 0.3f;
		d_params.draw_detections = false;
		d_params.frame_stride = 1;
		yolo_.SetDynamicParams(d_params);

	}

	void SendImage()
	{

		yolo_.ImageCallback(img_,sequence_);
		sequence_++;
	}

	// Once an image is processed by YOLO this function will be called to
	// store the data.
	void ImageCallback(const cv::Mat& drawimg, const darknet_wrapper::common::BoundingBoxes& boxes, const int& seq)
	{
		boxes_ = boxes;
		processed_images_++;
	}

	void RunBenchmark()
	{


		std::chrono::duration<float> elapsed_seconds = std::chrono::duration<float>::zero();
		start_time_ =  std::chrono::system_clock::now();

		while (elapsed_seconds < std::chrono::seconds(run_time_))
		{
			elapsed_seconds = std::chrono::system_clock::now() - start_time_;
			SendImage();
			std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
		}


		stop_time_ = std::chrono::system_clock::now();
		elapsed_seconds = stop_time_ - start_time_;



		std::cout << "//////////////////////////////////////////////" << std::endl;

		std::cout << "Images Processed: " << processed_images_ << std::endl;
		std::cout << "Time duration: " << elapsed_seconds.count() << std::endl;
		std::cout << "Benchmark FPS: " << processed_images_/elapsed_seconds.count() << std::endl;
		std::cout << "Darknet FPS: " << boxes_.fps << std::endl;


		std::cout << "//////////////////////////////////////////////" << std::endl;



	}




	YoloObjectDetector yolo_;
	darknet_wrapper::common::BoundingBoxes boxes_;
	cv::Mat img_;

	std::chrono::time_point<std::chrono::system_clock> start_time_, stop_time_;
	int processed_images_ = 0;
	int sequence_ = 0;
	bool first_ = true;

	// Parameters to change
	int run_time_;          // In seconds
	int sleep_time_;        // In miliseconds





};
}


int main(int argc, char **argv) {

    darknet_wrapper::UtilBenchmark u;
    u.RunBenchmark();

    return 0;
}