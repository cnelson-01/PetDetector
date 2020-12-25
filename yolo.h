//
// Created by chrisn on 12/25/2020.
//

#ifndef PETDETECTOR_YOLO_H
#define PETDETECTOR_YOLO_H

#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class YoloDetector {

public:

    // remove unnecessary bounding boxes
    void remove_box(Mat &frame, const vector<Mat> &out);

// draw bounding boxes
    void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat &frame);

// get output layers
    vector<String> getOutputsNames(const Net &net);

    YoloDetector();

    void detectAndShow(cv::Mat image);

    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    int inpWidth;        // Width of network's input image
    int inpHeight;       // Height of network's input image
    String configuration;
    String model;
    vector<string> classes;
};


#endif //PETDETECTOR_YOLO_H
