#ifndef IOH_H_
#define IOH_H_

#include<string>
#include<opencv2/core/core.hpp>


class IOHelper
{
private:
    /* data */
    std::string datadir;
    std::string outdir;
    std::string im0Path;
    std::string im1Path;
    std::string calibPath;
    int ndisp;
    int height;
    int width;

public:
    IOHelper(/* args */);
    ~IOHelper();
    void setUp(std::string im0Path, std::string im1Path, int ndisp, std::string outdir);
    int readCalib(cv::Mat mats[2]);
    void readImage(cv::Mat& left_img, cv::Mat& right_img);
};
#endif