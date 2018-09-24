#include "darknet_wrapper/YoloObjectDetector.h"

#ifdef DARKNET_FILE_PATH
const std::string kDarknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_wrapper {

YoloObjectDetector::YoloObjectDetector(
    std::string labels_filename, 
    std::string params_filename,
    std::string config_filename,
    std::string weights_filename) {

    config_file_path_ = new char[config_filename.length() + 1];
    weights_file_path_ = new char[weights_filename.length() + 1];
    strcpy(config_file_path_, config_filename.c_str());
    strcpy(weights_file_path_, weights_filename.c_str());

    if (InitParameters(labels_filename, params_filename))
    {
        InitThreadQueue();

        // Load network.
        InitNetwork();

        // Start the thread scheduler
        scheduler_thread_ = std::thread(&YoloObjectDetector::ThreadScheduler,this);

    }

}

//-------------------------------------------------------------------------------------------------------------------------

YoloObjectDetector::~YoloObjectDetector() {

    {
        boost::unique_lock<boost::shared_mutex> node_running_mutex(mutex_node_running_);
        is_program_running_ = false;
    }

    // Ensure that the last images get processed. 
    if (scheduler_thread_.joinable())
        scheduler_thread_.join();


    delete [] config_file_path_;
    delete [] weights_file_path_;
    delete [] data_file_path_;

    // Deallocate the dynamic memory. 
    if (frames_init_)
        for (int i = 0; i < CPUs_; i++) {
            delete ipls_[i];
            // delete [] img_threads_[i].dets;
        }
    delete [] img_threads_;


    for (int i =0; i < num_classes_; i++)
        delete [] class_labels_[i];
    free(class_labels_);


    for (int i = 0; i < 8; i++)
        free(alphabet_[i]);
    free(alphabet_);

    free_network(net_);


    /////////////////////////////////////////////////////
    //    Memory leak testing
    /////////////////////////////////////////////////////

    // There are some memory leaks from the net. Below are my attempts to find
    // them. Ultimately it doesn't matter since the net is used until the end
    // and when a program is no longer running, all the memory used by it in the
    // stack and heap are freed. If you brave the venture of finding the leaks, 
    // you can see below where I have searched for them. Good luck, valiant knight!


    // free(net_->seen);        //
    // free(net_->t);  
    // free(net_->scales);
    // free(net_->steps);


    // for (int i = 0; i < net_->n; i++)
    // {
    //     cuda_free(net_->layers[i].temp_gpu);
    //     cuda_free(net_->layers[i].temp2_gpu);
    //     cuda_free(net_->layers[i].temp3_gpu);
    //     cuda_free(net_->layers[i].dh_gpu);
    //     cuda_free(net_->layers[i].hh_gpu);
    //     cuda_free(net_->layers[i].prev_cell_gpu);
    //     cuda_free(net_->layers[i].cell_gpu);
    //     cuda_free(net_->layers[i].i_gpu);
    //     cuda_free(net_->layers[i].g_gpu);
    //     cuda_free(net_->layers[i].o_gpu);
    //     cuda_free(net_->layers[i].c_gpu);
    //     cuda_free(net_->layers[i].dc_gpu);
    //     cuda_free(net_->layers[i].bias_m_gpu);
    //     cuda_free(net_->layers[i].scale_m_gpu);
    //     cuda_free(net_->layers[i].bias_v_gpu);
    //     cuda_free(net_->layers[i].scale_v_gpu);
    // }


    // free(net_->hierarchy->leaf);  //
    // free(net_->hierarchy->parent); //
    // free(net_->hierarchy->child);  //
    // free(net_->hierarchy->group);  //
    // // free(net_->hierarchy->name);  //
    // free(net_->hierarchy->group_size); //
    // free(net_->hierarchy->group_offset); //
    // free(net_->hierarchy);

    // free(net_->cost);         //


// #ifdef GPU 
//     if (net_->workspace)  cuda_free(net_->workspace);    //    
//     if (net_->delta_gpu)  cuda_free(net_->delta_gpu);
//     if (net_->output_gpu) cuda_free(net_->output_gpu);
// #else
//     if (net_->workspace) free(net_->workspace);    //
// #endif

//     if (net_->workspace) free(net_->workspace);    //

//     if (net_->delta)     free(net_->delta);
//     if (net_->output)    free(net_->output);


}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::InitThreadQueue() {

    // Allocate memory for the queue that holds the image and thread info
    img_threads_ = new ImgThread[CPUs_];
    for (int i =0; i < CPUs_; i++) {
        img_threads_[i].thread_available = true;
        img_threads_[i].has_img = false;
    }
}

//-------------------------------------------------------------------------------------------------------------------------

bool YoloObjectDetector::InitParameters(std::string labels_filename, std::string params_filename) {

    bool parameters_initialized = false;

    // Load parameters
    if (params_.Initialize(params_filename) &&  cl_params_.Initialize(labels_filename))
    {

        //
        // Get class labels and convert them from string to char**
        //
        std::vector<std::string> class_labels;
        cl_params_.GetParam("labels", class_labels,
                        std::vector<std::string>(0));
        num_classes_ = class_labels.size();
        class_labels_ = (char**) realloc((void*) class_labels_, (num_classes_ + 1) * sizeof(char*));
        
        for (int i = 0; i < num_classes_; i++) {
            class_labels_[i] = new char[class_labels[i].length() + 1];
            strcpy(class_labels_[i], class_labels[i].c_str());
        }

        //
        // Path to data folder and convert string to char*
        //
        std::string data_path;
        data_path = kDarknetFilePath_;
        data_path += "/data";
        data_file_path_ = new char[data_path.length() + 1];
        strcpy(data_file_path_, data_path.c_str());

        //
        // Get static parameters
        //

        // Get the max number of threads the user wants.
        params_.GetParam("num_threads", num_threads_, 3);

        //
        // Get dynamic parameters: 
        //      
        //  Dynamic parameters are placed in a special container
        //  that handles reading from and writing to by multiple threads

        // Get prediction threshold
        float thresh;
        params_.GetParam("threshold", thresh, 0.3);
        d_params_.SetThreshold(thresh);

        // Get draw_detections
        bool draw_detections;
        params_.GetParam("draw_detections", draw_detections, false);
        d_params_.SetDrawDetections(draw_detections);

        int frame_stride;
        params_.GetParam("frame_stride", frame_stride, 1);
        d_params_.SetFrameStride(frame_stride);


        // Set the number of processors used to min(available_processors, desired_processors)
        int available_processors = boost::thread::hardware_concurrency();
        CPUs_ = ((num_threads_ < available_processors) ? num_threads_ : available_processors );
        CPUs_ = ((CPUs_ < 1) ? 1 : CPUs_);

        parameters_initialized = true;
    } else {
        std::cout << "YoloObjectDetector: Could not initialize parameters. Shutting down program." << std::endl;
    }
    return parameters_initialized;
}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::InitNetwork() {
  
    // Load pictures of the alphabet. This is used to draw letters
    // onto the image.
    alphabet_ = load_alphabet_with_file(data_file_path_);

    // Load the weights and configuration file to create the
    // network
    net_ = load_network(config_file_path_, weights_file_path_, 0);
    set_batch_network(net_, 1);

    // Dimensions of net
    net_width_ = net_->w;
    net_height_ = net_->h;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::SetDynamicParams(const common::DynamicParams& params) {

    d_params_.SetThreshold(params.threshold);
    d_params_.SetDrawDetections(params.draw_detections);
    d_params_.SetFrameStride(params.frame_stride);

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::ImageCallback(const cv::Mat& img, int seq = 0) {

    if (ProcessImage()) {

        int thread_index = GetAvailableThreadForImage();

        // A thread is available
        if (thread_index >= 0) {

            if (!frames_init_) {
                InitFrames(img);
            }

            if (img_size_ != img.size())
                std::cout << "ERROR::YoloObjectDetector Image: " << seq << " is not the same size as the first image. Skipping Image." << std::endl;
            else
                SetImage(img, thread_index, seq);
        }
    }
}


//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::InitFrames(const cv::Mat& img) {

    img_size_ = img.size();
    IplImage* ipl_img = CvMatToCvIplImage(img);
    darknet_imgs_.clear();

    // Allocate memory for different frames that will be used. One
    // for each thread.    
    for (int i = 0; i < CPUs_; i++) {

        darknet_imgs_.push_back(ipl_to_image(ipl_img));
        darknet_letters_.push_back(letterbox_image(darknet_imgs_[0],net_->w, net_->h));
        ipls_.push_back(cvCreateImage(cvSize(darknet_imgs_[0].w, darknet_imgs_[0].h), IPL_DEPTH_8U, darknet_imgs_[0].c));
    }

    frames_init_ = true;

    // Get the initial start time to calculate fps.
    start_time_ = std::chrono::system_clock::now();

    delete ipl_img;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::ThreadScheduler() {

    const auto wait_duration = std::chrono::milliseconds(30);
    std::thread t[CPUs_];
    std::vector<int> threads_available;

    while (IsProgramRunning()) {

        // See if there are threads available with images to be processed
        threads_available.clear();
        threads_available = GetAvailableThreadsForProcessing();

        // If there are images to be processed and threads read
        // start them. 
        if (threads_available.size() != 0) {

            for (int i = 0; i < threads_available.size(); i++) {

                int thread_index = threads_available[i];

                // Make sure the thread is done
                if (t[thread_index].joinable())
                    t[thread_index].join();

                // Change the status of the thread to busy or occupied
                OccupyThread(thread_index);

                // Start thread
                t[thread_index] = std::thread(&YoloObjectDetector::Yolo, this, GetImage(thread_index), thread_index);
            }
        } else {
            // Three are either no images to process or threads available.
            // Sleep for a little while then check if an image is read
            // or a thread is available. 
            std::this_thread::sleep_for(wait_duration);
        }
    }

    // Make sure all threads are done.
    for (int i = 0; i < CPUs_; i++) {
        if (t[i].joinable())
            t[i].join();
    }

}


//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::Yolo(const cv::Mat& img, int thread_index) {
    

    // Detections
    detection *dets = NULL;
    int nboxes = 0;

    // Format the image to the type that Darknet YOLO
    // uses. 
    FormatImage(img, thread_index);

    // Peform YOLO on the image to detect the objects
    DetectObjects(img, dets, nboxes, thread_index);

    // Increment the number of frames processed by 1
    IncNumFramesProcessed();

    // Draw detections and save it to draw_img
    if (d_params_.GetDrawDetections())
        DrawDetections(dets,nboxes,thread_index);

    // Organizes the bounding boxes in a nice way
    FormBoundingBoxes(img, dets, nboxes,thread_index);

    // Free up the memory
    free_detections(dets, nboxes);

    SendData(thread_index);

    // Indicate that the thread is done processing. 
    ReleaseThread(thread_index);

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::FormatImage(const cv::Mat& img, int thread_index) {

    IplImage* ipl_img = CvMatToCvIplImage(img);

    // Format IPL image to darknet_img_
    ipl_into_image(ipl_img, darknet_imgs_[thread_index]);

    // convert rgb to bgr
    rgbgr_image(darknet_imgs_[thread_index]);

    // Resize and copies the image
    letterbox_image_into(darknet_imgs_[thread_index], net_width_, net_height_, darknet_letters_[thread_index]);

    delete  ipl_img;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::DetectObjects(const cv::Mat& img, detection *&dets, int &nboxes, int thread_index) {
  
    // Since there is only one net, the net must only be accessed by 
    // one thread at a time. 

    // Get predictions
    layer l = net_->layers[net_->n - 1];
    float *X = darknet_letters_[thread_index].data;

    mutex_net_.lock();

    // Since there is only one net, the net must only be accessed by 
    // one thread at a time. 
    float *prediction = network_predict(net_, X);

    // Get bounding boxes for each prediction
    // Uses yolo detections
    dets = get_network_boxes(net_, darknet_imgs_[thread_index].w, darknet_imgs_[thread_index].h, d_params_.GetThreshold(), tree_threshold_, 0, 1, &nboxes);
    mutex_net_.unlock();

    // Used to remove multiple bounding boxes that encase the same object.
    if (nms_ > 0) do_nms_obj(dets, nboxes, l.classes, nms_);


}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::DrawDetections(detection *dets, int nboxes, int thread_index) {

    // Draw detections on the image
    // This method also prints the objects detected with 
    // their probability. 
    draw_detections(darknet_imgs_[thread_index], dets, nboxes, d_params_.GetThreshold(), class_labels_, alphabet_, num_classes_);

    // Formats darknet_img_ to ipl Image
    ImageToIpl(darknet_imgs_[thread_index], ipls_[thread_index]);

    // Convert from IplImage to cv::Mat
    img_threads_[thread_index].draw_img = cv::cvarrToMat(ipls_[thread_index]);
}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::FormBoundingBoxes(const cv::Mat& img, detection* dets, int& nboxes,int thread_index) {

    // Extract the bounding boxes and send them to ROS
    int i, j;
    int count = 0;
    int num_boxes = 0;
    for (i = 0; i < nboxes; ++i) {


        // Make sure that the dimensions of the bounding box
        // are within the frame's dimensions.

        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0)
            xmin = 0;
        if (ymin < 0)
            ymin = 0;
        if (xmax > 1)
            xmax = 1;
        if (ymax > 1)
            ymax = 1;

        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float bounding_box_width = xmax - xmin;
        float bounding_box_height = ymax - ymin;

        // define bounding box
        // bounding_box must be 1% size of frame (3.2x2.4 pixels)
        if (bounding_box_width > 0.01 && bounding_box_height > 0.01) {

             // Delcare bounding box
            common::BoundingBox bounding_box;
            common::Prediction prediction;

            // Since each bounding box can have multiple classes with
            // a probability higher than the threshold, 
            for (j = 0; j < num_classes_; ++j) {


                if (dets[i].prob[j]) {

                    prediction.class_label = class_labels_[j];
                    prediction.probability = dets[i].prob[j];
                    bounding_box.predictions.push_back(prediction);   
                }
            }

            if (bounding_box.predictions.size() > 0) {

                num_boxes++;
                bounding_box.xcent = x_center*img.cols;
                bounding_box.ycent = y_center*img.rows;
                bounding_box.xmin = xmin*img.cols;
                bounding_box.xmax = xmax*img.cols;
                bounding_box.ymin = ymin*img.rows;
                bounding_box.ymax = ymax*img.rows;
                img_threads_[thread_index].bounding_boxes.boxes.push_back(bounding_box);
            }           
        }
    }
   

    img_threads_[thread_index].bounding_boxes.stop_time = std::chrono::system_clock::now();

    img_threads_[thread_index].bounding_boxes.img_w = img.cols;
    img_threads_[thread_index].bounding_boxes.img_h = img.rows;
    img_threads_[thread_index].bounding_boxes.num_boxes = num_boxes;
    std::chrono::duration<float> elapsed_seconds = img_threads_[thread_index].bounding_boxes.stop_time - img_threads_[thread_index].bounding_boxes.start_time;
    img_threads_[thread_index].bounding_boxes.process_time = elapsed_seconds.count();
    img_threads_[thread_index].bounding_boxes.fps = GetFps();

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::SendData(int thread_index)
{

    // TODO::   Should I clone the image instead of passing a pointer to it?
    boost::lock_guard<boost::mutex> lock(send_mutex_);
    {
    if (fn_ptr_)
        fn_ptr_(
            img_threads_[thread_index].draw_img,
            img_threads_[thread_index].bounding_boxes,
            img_threads_[thread_index].bounding_boxes.seq);
    else
        std::cout << "YoloObjectDetector: You have not Subscribed to the data." << std::endl;
    }
}


//-------------------------------------------------------------------------------------------------------------------------

IplImage* YoloObjectDetector::CvMatToCvIplImage(const cv::Mat& img) {

    // IplImage is a cv function to change the format to 
    // Intel Image Processing image format.   
    return new IplImage(img);
}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::ImageToIpl(const image& darknet_img, IplImage *cv_img) {

    int x,y,k;
    if(darknet_img.c == 3) rgbgr_image(darknet_img);
    int step = cv_img->widthStep;

    for(y = 0; y < darknet_img.h; ++y){
        for(x = 0; x < darknet_img.w; ++x){
            for(k= 0; k < darknet_img.c; ++k){
                cv_img->imageData[y*step + x*darknet_img.c + k] = (unsigned char)(GetPixel(darknet_img,x,y,k)*255);
            }
        }
    }

}

//-------------------------------------------------------------------------------------------------------------------------

float YoloObjectDetector::GetPixel(image m, int x, int y, int c) {

  assert(x < m.w && y < m.h && c < m.c);
  return m.data[c*m.h*m.w + y*m.w + x];
}



//-------------------------------------------------------------------------------------------------------------------------

//
//                   Functions for Threading
//

//-------------------------------------------------------------------------------------------------------------------------

bool YoloObjectDetector::ProcessImage() {

    bool process = (frame_count_ % d_params_.GetFrameStride()) == 0;
    frame_count_++;
    return process;
}

//-------------------------------------------------------------------------------------------------------------------------

bool YoloObjectDetector::IsProgramRunning() {

   boost::shared_lock<boost::shared_mutex> node_running_mutex(mutex_node_running_);
   return is_program_running_;

}

//-------------------------------------------------------------------------------------------------------------------------

int YoloObjectDetector::GetAvailableThreadForImage() {

    boost::shared_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);

    // Return the index of the first thread that is available. 
    for (int i = 0; i < CPUs_; i++) {
        if (img_threads_[i].thread_available && !img_threads_[i].has_img) {
            return i;
        }
    }

    return -1;
}

//-------------------------------------------------------------------------------------------------------------------------

std::vector<int> YoloObjectDetector::GetAvailableThreadsForProcessing() {

    std::vector<int> available_threads;

    boost::shared_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);

    {
    // Return all of the indexs of the available threads. 
    for (int i = 0; i < CPUs_; i++) {
        if (img_threads_[i].thread_available && img_threads_[i].has_img)
        {
            available_threads.push_back(i);
        }
    }

    }

    return available_threads;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::SetImage(const cv::Mat& img, int thread_index, int seq) {

    boost::unique_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);

    img_threads_[thread_index].has_img = true;
    img_threads_[thread_index].img = img.clone();
    img_threads_[thread_index].bounding_boxes.boxes.clear();
    img_threads_[thread_index].bounding_boxes.start_time = std::chrono::system_clock::now();
    img_threads_[thread_index].bounding_boxes.seq = seq;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::ReleaseThread(int thread_index) {

    boost::unique_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);
    img_threads_[thread_index].has_img = false;
    img_threads_[thread_index].thread_available = true;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::OccupyThread(int thread_index) {

    boost::unique_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);
    img_threads_[thread_index].thread_available = false;

}

//-------------------------------------------------------------------------------------------------------------------------

cv::Mat& YoloObjectDetector::GetImage(int thread_index) {

    boost::shared_lock<boost::shared_mutex> img_thread_mutex(mutex_img_threads_);
    return img_threads_[thread_index].img;

}

//-------------------------------------------------------------------------------------------------------------------------

void YoloObjectDetector::IncNumFramesProcessed() {

    boost::unique_lock<boost::shared_mutex> fps_mutex(mutex_img_threads_);
    num_frames_processed_++;
}

//-------------------------------------------------------------------------------------------------------------------------

float YoloObjectDetector::GetFps() {

    boost::shared_lock<boost::shared_mutex> fps_mutex(mutex_img_threads_);
    std::chrono::duration<float> elapsed_seconds = std::chrono::system_clock::now() - start_time_;
    return num_frames_processed_/elapsed_seconds.count();
}

} /* namespace darknet_wrapper*/
