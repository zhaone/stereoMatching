#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include "mbp.hpp"

using namespace cv;
using namespace std;

int get_min_idx_m(Mat &message, int disp)
{
    float bestValue = 100000000;
    int bestIdx = 0;
    for (int d = 0; d < disp; d++)
    {
        float tmp = message.at<float>(d, 0);
        if (tmp < bestValue)
        {
            bestValue = tmp;
            bestIdx = d;
        }
    }
    return bestIdx;
}

MBP::~MBP() {}

MBP::MBP(Mat &aleftImg, Mat &arightImg, const int ndisp, const float smoothLambda, const float acostLambda, int aiter)
{
    // TODO convert to float?
    cv::Mat leftImg, rightImg;
    aleftImg.convertTo(leftImg, CV_32FC1, 1 / 255.0);
    arightImg.convertTo(rightImg, CV_32FC1, 1 / 255.0);
    height = leftImg.rows;
    width = leftImg.cols;
    disp = ndisp;
    costLambda = acostLambda;
    iter = aiter;

    cout << "height:" << height << ", width:"
         << width << ", img size:" << leftImg.size() << endl;

    int pb = disp + 2;
    cv::copyMakeBorder(leftImg, leftPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(rightImg, rightPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);

    smoothCostMat = Mat::zeros(cv::Size(disp, disp), CV_32FC1);

    msg.resize(height);
    obs.resize(height);
    for (int h = 0; h < height; h++)
    {
        msg[h].resize(width);
        obs[h].resize(width);
    }

    int NumThread = 4;
    int sh, eh, lineEveryThread = height / NumThread;
    vector<thread> tpool(NumThread);
    for (int tid = 0; tid < NumThread; tid++)
    {
        sh = tid * lineEveryThread;
        eh = tid == NumThread - 1 ? height : (tid + 1) * lineEveryThread;
        thread t(&MBP::calculateDataCostThread, this, sh, eh);
        tpool[tid] = move(t);
    }

    for (int tid = 0; tid < NumThread; tid++)
    {
        tpool[tid].join();
    }
    // cout << "----------------msg[200][200]-------------------" << endl << msg[200][200] << endl;
    // cout << "----------------obs[200][200]-------------------" << endl << obs[200][200] << endl;
    for (int d1 = 0; d1 < disp; d1++)
    {
        for (int d2 = 0; d2 < disp; d2++)
        {
            float diff = float(abs(d1 - d2));
            smoothCostMat.at<float>(d1, d2) = smoothLambda * (log(diff + 1));
        }
    }
    // cout << "----------------smoothCostMat-------------------" << endl << smoothCostMat << endl;
    // cout << "init finished" << endl;
}

void MBP::calculateDataCostThread(int sh, int eh)
{
    for (int h = sh; h < eh; h++)
    {
        for (int w = 0; w < width; w++)
        {
            msg[h][w] = Mat::zeros(cv::Size(4, disp), CV_32FC1);

            Mat tmp(cv::Size(1, disp), CV_32FC1);
            for (int d = 0; d < disp; d++)
            {
                tmp.at<float>(d, 0) = calculateDataCost(h, w, d);
            }
            obs[h][w] = costLambda * tmp / sum(tmp);
        }
    }
}

float MBP::calculateDataCost(const int h, const int w, const int d)
{
    int ph = disp + h;
    int pw = disp + w;
    int radius = 2;
    int dim = 2 * radius + 1;

    // Rect
    cv::Mat w1 = leftPaddingImg(cv::Rect(pw, ph, dim, dim));
    cv::Mat w2 = rightPaddingImg(cv::Rect(pw - d, ph, dim, dim));

    cv::Mat absDiff;
    cv::absdiff(w1, w2, absDiff);

    return float(sum(absDiff.mul(absDiff))[0]);
}

void MBP::beliefPropagateThread(vector<vector<Mat>> &msgCopy, int sh, int eh)
{
    for (int dir = 0; dir < 4; dir++)
    {
        cout << "dir: " << dir << endl;
        // for (int h = 1; h < height - 1; h++)
        for (int h = sh; h < eh; h++)
        {
            for (int w = 1; w < width - 1; w++)
            {
                switch (dir)
                {
                case 0: //left
                    maxProduct(msgCopy, h, w, dir).copyTo(msg[h][w - 1].col(1));
                    break;
                case 1: //right
                    maxProduct(msgCopy, h, w, dir).copyTo(msg[h][w + 1].col(0));
                    break;
                case 2: //up
                    maxProduct(msgCopy, h, w, dir).copyTo(msg[h - 1][w].col(3));
                    break;
                case 3: //down
                    maxProduct(msgCopy, h, w, dir).copyTo(msg[h + 1][w].col(2));
                    break;
                default:;
                }
            }
        }
    }
}

void MBP::beliefPropagate(bool visualize = false)
{
    for (int i = 0; i < iter; i++)
    {
        cout << "iter " << i << " start" << endl;
        // copy msg
        cout << "copy message" << endl;
        vector<vector<Mat>> msgCopy(height);
        for (int h = 0; h < height; h++)
        {
            msgCopy[h].resize(width);
            for (int w = 0; w < width; w++)
                msgCopy[h][w] = msg[h][w].clone();
        }
        // pass
        int NumThread = 4;
        int sh, eh, lineEveryThread = height / NumThread;
        vector<thread> tpool(NumThread);
        for (int tid = 0; tid < NumThread; tid++)
        {
            sh = tid == 0 ? 1 : tid * lineEveryThread;
            eh = tid == NumThread - 1 ? height - 1 : (tid + 1) * lineEveryThread;
            thread t(&MBP::beliefPropagateThread, this, ref(msgCopy), sh, eh);
            tpool[tid] = move(t);
        }

        for (int tid = 0; tid < NumThread; tid++)
        {
            tpool[tid].join();
        }

        if (visualize)
        {
            Mat iterMap = getDispMap();
            cv::imshow("debug", iterMap);
            cv::waitKey(0);
        }
    }
}

Mat MBP::maxProduct(vector<vector<Mat>> &msgCopy, int h, int w, int dir)
{
    double minVal;
    Mat res(cv::Size(1, disp), CV_32FC1);
    for (int d = 0; d < disp; d++)
    {
        Mat message = smoothCostMat.col(d) + obs[h][w];
        for (int n = 0; n < dir; n++)
        {
            if (n != dir)
            {
                message += msgCopy[h][w].col(n);
            }
        }
        message += obs[h][w];
        minMaxIdx(message, &minVal);
        res.at<float>(d, 0) = minVal;
        // smooth cost !!!!
    }
    return res;
}

Mat MBP::getDispMap()
{
    int minIdx = 0;
    Mat DispMap(cv::Size(width, height), CV_8UC1);
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // cout<< h << ", " << w <<endl;
            Mat sumMsg(Size(1, disp), CV_32FC1);
            reduce(msg[h][w], sumMsg, 1, REDUCE_SUM, CV_32FC1);
            sumMsg += obs[h][w];
            // minMaxIdx(sumMsg, &minVal, &maxVal, &minIdx, &maxIdx);
            DispMap.at<uchar>(h, w) = get_min_idx_m(sumMsg, disp);
        }
    }
    return DispMap;
}

Mat MBP::do_match()
{
    beliefPropagate(true);
    return getDispMap();
}