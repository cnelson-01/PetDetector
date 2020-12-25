//
// Created by chrisn on 12/25/2020.
//

#include "yolo.h"

void YoloDetector::detectAndShow(cv::Mat frame) {
    // read the image
//    frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    Mat blob;
    Net net = readNetFromDarknet(configuration, model);

    // convert image to blob
    blobFromImage(frame, blob, 1 / 255, cv::Size(frame.cols, frame.rows), Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    remove_box(frame, outs);

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

    Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    static const string kWinName = "Deep learning object detection in OpenCV";

    imshow(kWinName, frame);
}

YoloDetector::YoloDetector() {
    confThreshold = 0.5; // Confidence threshold
    nmsThreshold = 0.4;  // Non-maximum suppression threshold
    inpWidth = 416;        // Width of network's input image
    inpHeight = 416;       // Height of network's input image

// get labels of all classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // load model weights and architecture
    configuration = "yolov3.cfg";
    model = "yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(configuration, model);
    Mat frame, blob;
};


void YoloDetector::remove_box(Mat &frame, const vector<Mat> &outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        draw_box(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

// Draw the predicted bounding box
void YoloDetector::draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat &frame) {
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)),
              Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<Mat> outs;

vector<String> YoloDetector::getOutputsNames(const Net &net) {
    static vector<String> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}