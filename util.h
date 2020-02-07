#ifndef UTIL_H
#define UTIL_H

#include<opencv2/opencv.hpp>
#include<string>

double countTime()
{
    return static_cast<double>(clock());
}

void showDispMap(const std::string name, cv::Mat &dispMap, const int maxDisp, bool save=false)
{
    dispMap *= 256/maxDisp;
    cv::imshow(name, dispMap);
    if(save)
    {
        cv::imwrite(name + ".png", dispMap);
    }
    cv::waitKey(0);
}
#endif
