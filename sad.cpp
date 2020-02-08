#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "sad.hpp"

using namespace std;
using namespace cv;

Mat SAD::do_match(Mat &aleftImage, Mat &arightImage)
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

    // Mat dispMap(height, width, CV_8UC1);
    Mat dispMap(height, width, CV_16UC1);

    int dim = 2 * radius + 1;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            int ph = disp + h;
            int pw = disp + w;

            cv::Mat w1 = leftPaddingImg(cv::Rect(pw, ph, dim, dim));

            int bestIdx = 0;
            float maxVal = 10000000;
            for (int d = 0; d < disp; d++)
            {
                // Rect
                cv::Mat w2 = rightPaddingImg(cv::Rect(pw - d, ph, dim, dim));
                Mat Dif;
                absdiff(w1, w2, Dif);
                Scalar ADD = sum(Dif);
                float tmp = ADD[0];

                if (tmp < maxVal)
                {
                    maxVal = tmp;
                    bestIdx = d;
                }
            }
            // dispMap.at<uchar>(h, w) = bestIdx;
            dispMap.at<ushort>(h, w) = bestIdx;
        }
    }
    return dispMap;
}