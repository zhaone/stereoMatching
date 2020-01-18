
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iomanip>

class SAD
{
public:
    SAD() : winSize(7), DSR(30) {}
    SAD(int _winSize, int _DSR) : winSize(_winSize), DSR(_DSR) {}
    cv::Mat do_match(cv::Mat &L, cv::Mat &R); 
private:
    int winSize; //kernel size
    int DSR;     //search window size
};