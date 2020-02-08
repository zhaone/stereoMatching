// see https://blog.csdn.net/liulina603/article/details/53302168
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "ncc.hpp"

using namespace std;
using namespace cv;

Mat NCC::do_match(Mat &aleftImage, Mat &arightImage)
{
    int height = aleftImage.rows;
    int width = aleftImage.cols;
    Mat leftImg, rightImg;
    cout << "image shape: " << height << "," << width << endl;
    aleftImage.convertTo(leftImg, CV_32FC1, 1 / 255.0);
    arightImage.convertTo(rightImg, CV_32FC1, 1 / 255.0);

    int pb = disp + radius;
    cv::Mat leftPaddingImg, rightPaddingImg;
    cv::copyMakeBorder(leftImg, leftPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(rightImg, rightPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);

    Mat dispMap(height, width, CV_16UC1);

    int dim = 2 * radius + 1;
    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int ph = disp + h;
            int pw = disp + w;

            cv::Mat w1 = leftPaddingImg(cv::Rect(pw, ph, dim, dim));
            cv::Scalar mean1 = cv::mean(w1);
            cv::Mat sub1 = w1 - mean1;

            int bestIdx = 0;
            float maxVal = -10000000;
            for(int d = 0; d < disp; d++)
            {
                // Rect
                cv::Mat w2 = rightPaddingImg(cv::Rect(pw - d, ph, dim, dim));
                cv::Scalar mean2 = cv::mean(w2);
                cv::Mat sub2 = w2 - mean2;

                cv::Scalar numerator = cv::sum(sub1.mul(sub2));
                double denominator = cv::norm(w1) * cv::norm(w2);
                float tmp = (numerator / denominator)[0];
                

                if (tmp > maxVal)
                {
                    maxVal = tmp;
                    bestIdx = d;
                }
            }
            dispMap.at<ushort>(h, w) = bestIdx;
        }
    }
    return dispMap;
}