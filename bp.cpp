#include <iostream>
#include <cmath>
#include "bp.hpp"

BP::BP(const int ndisp, const float lambda, const float smoothnessParam, const int aradius)
{
    // set all the fields
    disp = ndisp;
    radius = aradius;
    // compute PHI
    PHI = cv::Mat::zeros(cv::Size(disp, disp), CV_32FC1);
    for (int i = 0; i < disp; i++)
        for (int j = 0; j < disp; j++)
            PHI.at<float>(i, j) = abs(i - j) < smoothnessParam ? abs(i - j) : smoothnessParam;
    PHI *= lambda;
    cv::Mat tmp = cv::Mat(disp, disp, CV_32FC1);
}

float BP::matchLocalProb(cv::Mat &leftPaddingImg, cv::Mat &rightPaddingImg, const int h, const int w, const int d)
{
    int ph = disp + h;
    int pw = disp + w;
    int dim = 2*radius + 1;

    cv::Mat w1 = leftPaddingImg(cv::Rect(pw, ph, dim, dim));
    cv::Mat w2 = rightPaddingImg(cv::Rect(pw + d, ph, dim, dim));

    cv::Mat absDiff;
    cv::absdiff(w1, w2, absDiff);
    float cost = cv::mean(absDiff)[0];

    // return (1-ep) * std::exp(-cost/dd) + ep;
    return cost;
}

void BP::init(cv::Mat &leftImage, cv::Mat &rightImage)
{
    // acutally, not all edge need so large padding
    int pb = disp + radius;
    cv::Mat leftPaddingImg, rightPaddingImg;
    cv::copyMakeBorder(leftImage, leftPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(rightImage, rightPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);

    msg.resize(height);
    float element = cv::log(float(disp));

    for (int i = 0; i < height; i++)
    {
        msg[i].resize(width);
        for (int j = 0; j < width; j++)
        {
            // msg[i][j] = 1 / float(disp) * cv::Mat::ones(cv::Size(5, disp), CV_32FC1);
            msg[i][j] = element * cv::Mat::zeros(cv::Size(5, disp), CV_32FC1);

            for (int k = 0; k < disp; k++)
            {
                msg[i][j].at<float>(k, 4) = matchLocalProb(leftPaddingImg, rightPaddingImg, i, j, k);
            }
        }
    }
}

void BP::do_bp(int iter)
{
    double minVal;
    while(iter > 0)
    {
        std::cout << "inter: " << iter << std::endl;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                cv::Mat msgCopy = msg[h][w].clone();
                for(int nei = 0; nei < 4; nei++)
                {
                    for (int d = 0; d < disp; d++)
                    {
                        cv::Mat tmp = PHI.col(d);
                        for (int idx = 0; idx < 5; idx++)
                        {
                            if (idx != nei)
                            {
                                tmp += msgCopy.col(idx);
                            }
                        }
                        cv::minMaxLoc(tmp, &minVal, NULL);
                        msg[h][w].at<float>(d, nei) = minVal;
                    }
                }
            }
        }
        iter--;
    }
}

cv::Mat BP::belief()
{
    std::vector<std::vector<cv::Mat>> beliefMat(height);
    for (int i = 0; i < height; i++)
        beliefMat[i].resize(width);
    
    cv::Mat disparity(cv::Size(height, width), CV_8UC1);

    int minIdx, maxIdx;

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // std::cout << "msg[h][w]" << msg[h][w] << std::endl;
            beliefMat[h][w] = cv::Mat::zeros(cv::Size(1, disp), CV_32FC1);
            for (int idx = 0; idx < 4; idx++)
                beliefMat[h][w] += msg[h][w].col(idx);
            // std::cout << "beliefMat[h][w]" << beliefMat[h][w] << std::endl;
            cv::minMaxIdx(beliefMat[h][w], NULL, NULL, &minIdx, &maxIdx);
            // std::cout << "minIdx " << minIdx << std::endl;
            disparity.at<uchar>(w, h) = minIdx * 256 / disp;
        }
    }
    return disparity;
}

cv::Mat BP::do_match(cv::Mat &leftImage, cv::Mat &rightImage, int iter)
{
    height = leftImage.rows;
    width = leftImage.cols;
    std::cout << "img height: " << height << std::endl;
    std::cout << "img width: " << width << std::endl;
    std::cout << leftImage.size() << std::endl;
    init(leftImage, rightImage);
    do_bp(iter);
    std::cout << "belief" << std::endl;
    return belief();
}