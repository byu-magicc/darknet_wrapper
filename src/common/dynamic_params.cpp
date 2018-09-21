#include "darknet_core/common/dynamic_params.h"


namespace yolo_core { namespace common {

void TDynamicParams::SetThreshold(float threshold) 
{
	boost::unique_lock<boost::shared_mutex> lock(threshold_mutex_);
	if( threshold != -999.0f)
		d_params_.threshold = std::min(std::max(threshold, 0.0f), 1.0f); 

}

float TDynamicParams::GetThreshold() 
{
	boost::shared_lock<boost::shared_mutex> lock(threshold_mutex_);
	return d_params_.threshold;
}

void TDynamicParams::SetDrawDetections(bool draw_detections) 
{
	boost::unique_lock<boost::shared_mutex> lock(draw_mutex_);
	d_params_.draw_detections = draw_detections;
}

bool TDynamicParams::GetDrawDetections() 
{
	boost::shared_lock<boost::shared_mutex> lock(draw_mutex_);
	return d_params_.draw_detections;
}


void TDynamicParams::SetFrameStride(int frame_stride) 
{
	boost::unique_lock<boost::shared_mutex> lock(frame_mutex_);
	if (frame_stride != -999)
		d_params_.frame_stride = std::max(frame_stride,1);
}


int TDynamicParams::GetFrameStride() 
{
	boost::shared_lock<boost::shared_mutex> lock(frame_mutex_);
	return d_params_.frame_stride;

}
}}