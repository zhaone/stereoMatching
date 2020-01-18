#include <iostream>
#include <opencv2/opencv.hpp>
#include <iomanip>

class NCC
{
public:
    NCC() : winSize(7), DSR(30) {}
    NCC(int _winSize, int _DSR) : winSize(_winSize), DSR(_DSR) {}
    cv::Mat do_match(cv::Mat &leftImage, cv::Mat &rightImage); 
private:
    int winSize; //kernel size
    int DSR;     //search window size
};