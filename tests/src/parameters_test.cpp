#include <gtest/gtest.h>
#include <darknet_wrapper/common/dynamic_params.h>
#include <darknet_wrapper/YoloObjectDetector.h>

#ifdef WRAPPER_FILE_PATH
const std::string kTestsDirPath = WRAPPER_FILE_PATH;

	std::string labels_filename = kTestsDirPath + "/yolo_network_config/labels/yolov3-tiny.yaml";
	std::string config_filename = kTestsDirPath + "/yolo_network_config/cfg/yolov3-tiny.cfg";
	std::string weights_filename = kTestsDirPath + "/yolo_network_config/weights/yolov3-tiny.weights";
	std::string params_filename = kTestsDirPath + "/tests/param/test_params.yaml";

#else
#error Path of WRAPPER_FILE_PATH repository is not defined in CMakeLists.txt.
#endif

class ExposeMembers : public darknet_wrapper::YoloObjectDetector {

public:
	ExposeMembers(): YoloObjectDetector(
    labels_filename, 
    params_filename,
    config_filename,
    weights_filename) {}

    ~ExposeMembers()=default;

    char** GetLabels() {return class_labels_;}
    int GetNumClasses(){return num_classes_;}
    int GetNumCPUs() {return CPUs_;}
    int GetNumThreads() {return num_threads_;}
    float GetThreshold() {return d_params_.GetThreshold();}
    bool GetDrawDetections() {return d_params_.GetDrawDetections();}
    int GetFrameStride() {return d_params_.GetFrameStride();}


    void SetDParams(const darknet_wrapper::common::DynamicParams& params)
    {
    	SetDynamicParams(params);
    }

};

// Tests static parameters and labels
TEST (ParamTest, StaticLabelsParameters) {

	ExposeMembers p;


	EXPECT_EQ(p.GetNumThreads(),3);

	// See if labels are loaded properly
	char** labels = p.GetLabels();
	EXPECT_EQ(p.GetNumClasses(),80);
	EXPECT_STREQ(labels[0],"person");
	EXPECT_STREQ(labels[79],"toothbrush");


}


// TEST (ParamTest, DynamicParameters) {

// 	ExposeMembers p;

// 	// Test the default dynamic parameters
// 	EXPECT_EQ(p.GetThreshold(),0.3f);
// 	EXPECT_EQ(p.GetDrawDetections(),false);
// 	EXPECT_EQ(p.GetFrameStride(),1);

// 	// test case when values are not set
// 	darknet_wrapper::common::DynamicParams d_params;
// 	p.SetDParams(d_params);
// 	EXPECT_EQ(p.GetThreshold(),0.3f);
// 	EXPECT_EQ(p.GetDrawDetections(),false);
// 	EXPECT_EQ(p.GetFrameStride(),1);



// 	// Change the parameters
// 	d_params.threshold = 0.6f;
// 	d_params.draw_detections = true;
// 	d_params.frame_stride = 2;
// 	p.SetDParams(d_params);

// 	// Test the changed parameters
// 	EXPECT_EQ(p.GetThreshold(),0.6f);
// 	EXPECT_EQ(p.GetDrawDetections(),true);
// 	EXPECT_EQ(p.GetFrameStride(),2);

// 	// Test saturation for threshold and frame stride
// 	d_params.threshold = 10;
// 	p.SetDParams(d_params);
// 	EXPECT_LE(p.GetThreshold(),1);
// 	d_params.threshold = -50;
// 	d_params.frame_stride = -2;
// 	p.SetDParams(d_params);
// 	EXPECT_GE(p.GetThreshold(),0);
// 	EXPECT_GE(p.GetFrameStride(),1);

// }





int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
