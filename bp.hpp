#ifndef __BP_H_
#define __BP_H_

#include <vector>
#include <opencv2/opencv.hpp>

class BP
{
private:
    // params
    int disp;
    int radius;
    cv::Mat PHI;
    // image size
    int width;
    int height;
    // messages
    // std::vector<std::vector<cv::Mat>> neiMsg; // height*weight*4*disp
    // std::vector<std::vector<cv::Mat>> obsMsg; // height*weight*disp*1
    std::vector<std::vector<cv::Mat>> msg; // height*weight*4*disp, last one is y_s

public:
    BP(const int ndisp, const float alambda, const float smoothnessParam, const int aradius);
    float matchLocalProb(cv::Mat &leftImage, cv::Mat &rightImage, const int x, const int y, const int d);
    void init(cv::Mat &leftImage, cv::Mat &rightImage);
    void do_bp(const int iter);
    cv::Mat belief();
    cv::Mat do_match(cv::Mat &leftImage, cv::Mat &rightImage, int iter);
};

#endif