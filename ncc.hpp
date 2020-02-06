#include <iostream>
#include <opencv2/opencv.hpp>
#include <iomanip>

class NCC
{
public:
    NCC() : radius(2), disp(30) {}
    NCC(int _radius, int _disp) : radius(_radius), disp(_disp) {}
    cv::Mat do_match(cv::Mat &leftImage, cv::Mat &rightImage); 
private:
    int radius;  //kernel size
    int disp;     //search window size
};