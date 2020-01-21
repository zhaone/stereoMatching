// see https://blog.csdn.net/aaronmorgan/article/details/79121434
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "ncc.hpp"

using namespace std;
using namespace cv;

Mat NCC::do_match(Mat &leftImage, Mat &rightImage)
{
    int Height = leftImage.rows;
    int Width = leftImage.cols;
    cv::Mat L,R;
    leftImage.convertTo(L, CV_32FC1);
    rightImage.convertTo(R, CV_32FC1);
    cv::Mat Disparity(Height, Width, CV_8U, Scalar(0));
    
    for (int i = 0; i < Width - winSize; i++)
    {
        for (int j = 0; j < Height - winSize; j++)
        {
            cv::Mat Kernel_L = L(Rect(i, j, winSize, winSize));
            cv::Scalar meanL = cv::mean(Kernel_L);
            Kernel_L -= meanL;

            Mat MM(1, DSR, CV_32F, Scalar(0));

            for (int k = 0; k < DSR; k++)
            {
                int x = i - k;
                if (x >= 0)
                {
                    cv::Mat Kernel_R = R(Rect(x, j, winSize, winSize));
                    cv::Scalar meanR = cv::mean(Kernel_R);
                    Kernel_R -= meanR;

                    cv::Scalar numerator = cv::sum(Kernel_L.mul(Kernel_R));
                    cv::Scalar denominator = cv::norm(Kernel_L) * cv::norm(Kernel_R);
                    float a = (numerator/denominator)[0];
                    MM.at<float>(k) = a;
                }
            }
            Point minLoc;
            minMaxLoc(MM, NULL, NULL, &minLoc, NULL);

            int loc = minLoc.x;
            Disparity.at<uchar>(j, i) = loc*4;
        }
        double rate = double(i) / (Width);
    }
    return Disparity;
}