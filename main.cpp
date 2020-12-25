#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolo.h"

int main() {

    YoloDetector yolo;
    cv::VideoCapture cap;
    if (!cap.open(0)) {
        std::cout << "couldn't open video" << std::endl;
        return 0;
    }
    for (;;) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break; // end of video stream
        yolo.detectAndShow(frame);
        if (cv::waitKey(10) == 27) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}
