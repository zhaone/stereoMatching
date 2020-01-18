// see https://blog.csdn.net/liulina603/article/details/53302168
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "sad.hpp"

using namespace std;
using namespace cv;

Mat SAD::do_match(Mat &L, Mat &R)
{
    int Height = L.rows;
    int Width = L.cols;
    cout << "image shape: " << Height << "," << Width << endl;
    Mat Kernel_L(Size(winSize, winSize), CV_8U, Scalar::all(0));
    Mat Kernel_R(Size(winSize, winSize), CV_8U, Scalar::all(0));
    Mat Disparity(Height, Width, CV_8U, Scalar(0));

    for (int i = 0; i < Width - winSize; i++) //
    {
        for (int j = 0; j < Height - winSize; j++)
        {
            Kernel_L = L(Rect(i, j, winSize, winSize));
            Mat MM(1, DSR, CV_32F, Scalar(0)); //

            for (int k = 0; k < DSR; k++)
            {
                int x = i - k;
                if (x >= 0)
                {
                    Kernel_R = R(Rect(x, j, winSize, winSize));
                    Mat Dif;
                    absdiff(Kernel_L, Kernel_R, Dif); //
                    Scalar ADD = sum(Dif);
                    float a = ADD[0];
                    MM.at<float>(k) = a;
                }
            }
            Point minLoc;
            minMaxLoc(MM, NULL, NULL, &minLoc, NULL);

            int loc = minLoc.x;
            Disparity.at<char>(j, i) = loc * 16;
        }
        double rate = double(i) / (Width);
        // cout << "Process: " << rate * 100 << "%" << endl;
    }
    return Disparity;
}