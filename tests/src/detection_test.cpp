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
	std::string params_filename = kTestsDirPath + "/tests/params/test_params.yaml";

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


class Detection {

public:


Detection() : yolo(
labels_filename, 
params_filename,
config_filename,
weights_filename){

	yolo.Subscriber(&Detection::ImageCallback,this);

	// Change the parameters
	darknet_wrapper::common::DynamicParams d_params;
	d_params.threshold = 0.3f;
	d_params.draw_detections = true;
	d_params.frame_stride = true;
	yolo.SetDynamicParams(d_params);

}

~Detection() = default;

// Sends the images to Darknet Core
void SendImage(int index) {

	cv::Mat img;
	img = cv::imread(images_[index],CV_LOAD_IMAGE_COLOR);

	// Ensures that all images are the same size.
	cv::resize(img,img,cv::Size(768,576));

	if (img.data) {
		yolo.ImageCallback(img,index);

		std::cout << "image sent" << std::endl;
	}
	else
		std::cout << "Image not found!" << std::endl;
}


// Once an image is processed by YOLO this function will be called to
// store the data.
void ImageCallback(const cv::Mat& drawimg, const darknet_wrapper::common::BoundingBoxes& boxes, const int& seq)
{

	drawn_images_.push_back(drawimg.clone());
	boxes_.push_back(boxes);
	seq_.push_back(seq);


}

// Some bounding boxes will return multiple associated class labels.
// This function returns the label with the highest probability.
std::string GetBestLabel(int img_index, int box_index)
{

	int index = 0;
	float best_prediction = 0;
	for (int i =0; i < boxes_[img_index].boxes[box_index].predictions.size();i++)
	{
		if (boxes_[img_index].boxes[box_index].predictions[i].probability > best_prediction) {

			best_prediction = boxes_[img_index].boxes[box_index].predictions[i].probability;
			index = i;
		}
	}

	return boxes_[img_index].boxes[box_index].predictions[index].class_label;
}



std::vector<std::string> images_ = {dog_file,eagle_file,horses_file,person_file};
std::vector<darknet_wrapper::common::BoundingBoxes> boxes_;
std::vector<cv::Mat> drawn_images_;
std::vector<int> seq_;
int index_ = 0;

darknet_wrapper::YoloObjectDetector yolo;



};

class DetectionTest :public ::testing::Test {

public:
	Detection d;

};

// Tests detection and classification without threading
// by processing only one image at a time.
TEST_F(DetectionTest, BasicDetection) {

	// Sends the first image
	d.SendImage(0);
	int count = 0;

	// Sends the next image once the previous one is done.
	while (d.seq_.size() < d.images_.size())
	{

		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		std::cout << " still waiting: " << d.seq_.size() << std::endl;

		// Once an image is processed, the size of d.seq.size() will increase.
		// This allows us to use it to see if the image is done processing.
		if (d.seq_.size() > count) {
			count++;
			if (count < d.images_.size())
				d.SendImage(count);
		}

	}

	// Sequence tests
	EXPECT_EQ(0, d.seq_[0]);
	EXPECT_EQ(1, d.seq_[1]);
	EXPECT_EQ(2, d.seq_[2]);
	EXPECT_EQ(3, d.seq_[3]);




	// First image
	EXPECT_STREQ(d.GetBestLabel(0,0).c_str(), "car");
	EXPECT_STREQ(d.GetBestLabel(0,1).c_str(), "bicycle");
	EXPECT_STREQ(d.GetBestLabel(0,2).c_str(), "dog");

	cv::imshow("Drawn Images",d.drawn_images_[0]);
	cv::waitKey(0);

	// Second image
	EXPECT_STREQ(d.GetBestLabel(1,0).c_str(), "bird");

	cv::imshow("Drawn Images",d.drawn_images_[1]);
	cv::waitKey(0);

	// Third image
	EXPECT_STREQ(d.GetBestLabel(2,0).c_str(), "horse");
	EXPECT_STREQ(d.GetBestLabel(2,1).c_str(), "horse");
	EXPECT_STREQ(d.GetBestLabel(2,2).c_str(), "horse");

	cv::imshow("Drawn Images",d.drawn_images_[2]);
	cv::waitKey(0);

	// Fourth image
	EXPECT_STREQ(d.GetBestLabel(3,0).c_str(), "sheep");
	EXPECT_STREQ(d.GetBestLabel(3,1).c_str(), "person");
	EXPECT_STREQ(d.GetBestLabel(3,2).c_str(), "dog");

	cv::imshow("Drawn Images",d.drawn_images_[3]);
	cv::waitKey(0);

	cv::destroyWindow("Drawn Images");

}


TEST_F(DetectionTest, Threading) {

	bool sent_last = false;

	std::chrono::time_point<std::chrono::system_clock> start_time, stop_time;
	start_time =  std::chrono::system_clock::now();

	// Send the first three images since we are only using 3 threads.
	// Send the last image once a thread is available.
	d.SendImage(0);
	d.SendImage(1);
	d.SendImage(2);


	// wait until a thread is available before sending the last image.
	while (d.seq_.size() < d.images_.size())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (d.seq_.size() > 0 && !sent_last)
		{
			d.SendImage(3);
			sent_last = true;
		}
	}


	stop_time =  std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = stop_time - start_time;

    std::cout << std::endl << "############################" <<std::endl;
	std::cout << "# Estimated time: " << elapsed_seconds.count() << "s" << std::endl;
	std::cout << "############################" << std::endl;
	
	// Since threading is used, data can come in any order. Let's put
	// it back to the original order.
	std::vector<darknet_wrapper::common::BoundingBoxes> boxes_temp =  d.boxes_;
	for (int i=0; i < d.seq_.size(); i++) {

		boxes_temp[d.seq_[i]] = d.boxes_[i];
	}
	d.boxes_ = boxes_temp;


	//
	// First image
	//

	// First image comparisons
	EXPECT_STREQ(d.GetBestLabel(0,0).c_str(), "car");
	EXPECT_STREQ(d.GetBestLabel(0,1).c_str(), "bicycle");
	EXPECT_STREQ(d.GetBestLabel(0,2).c_str(), "dog");

	// Display image for visual inspections
	std::vector<int>::iterator itr = std::find(d.seq_.begin(), d.seq_.end(),0);
	int index = std::distance(d.seq_.begin(),itr);
	cv::imshow("Drawn Images",d.drawn_images_[index]);
	cv::waitKey(0);

	//
	// Second image
	//

	// Second image comparisons
	EXPECT_STREQ(d.GetBestLabel(1,0).c_str(), "bird");

	// Display image for visual inspections
	itr = std::find(d.seq_.begin(), d.seq_.end(),1);
	index = std::distance(d.seq_.begin(),itr);
	cv::imshow("Drawn Images",d.drawn_images_[index]);
	cv::waitKey(0);

	//
	// Third image
	//

	// Third image comparisons
	EXPECT_STREQ(d.GetBestLabel(2,0).c_str(), "horse");
	EXPECT_STREQ(d.GetBestLabel(2,1).c_str(), "horse");
	EXPECT_STREQ(d.GetBestLabel(2,2).c_str(), "horse");

	// Display image for visual inspections
	itr = std::find(d.seq_.begin(), d.seq_.end(),2);
	index = std::distance(d.seq_.begin(),itr);
	cv::imshow("Drawn Images",d.drawn_images_[index]);
	cv::waitKey(0);

	//
	// Fourth image
	//

	// Fourth image comparisons
	EXPECT_STREQ(d.GetBestLabel(3,0).c_str(), "sheep");
	EXPECT_STREQ(d.GetBestLabel(3,1).c_str(), "person");
	EXPECT_STREQ(d.GetBestLabel(3,2).c_str(), "dog");

	// Display image for visual inspections
	itr = std::find(d.seq_.begin(), d.seq_.end(),3);
	index = std::distance(d.seq_.begin(),itr);
	cv::imshow("Drawn Images",d.drawn_images_[index]);
	cv::waitKey(0);


	cv::destroyWindow("Drawn Images");

}

int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}