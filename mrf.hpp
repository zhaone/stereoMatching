#ifndef MRF_H_
#define MRF_H_

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

enum Direction
{
    Left,
    Right,
    Up,
    Down,
    Data
};

typedef float msg_t;
typedef float energy_t;
typedef float smoothness_cost_t;
typedef float data_cost_t;

struct MarkovRandomFieldNode
{
    msg_t *leftMessage;
    msg_t *rightMessage;
    msg_t *upMessage;
    msg_t *downMessage;
    msg_t *dataMessage;
    int bestAssignmentIndex;
};

struct MarkovRandomFieldParam
{
    int maxDisparity, iteration;
    float lambda, smoothnessParam;
};

struct MarkovRandomField
{
    vector<vector<MarkovRandomFieldNode>> grid;
    MarkovRandomFieldParam param;
    int height, width;
};

void initializeMarkovRandomField(MarkovRandomField &mrf, Mat leftImg, Mat rightImg, MarkovRandomFieldParam param);
void sendMsg(MarkovRandomField &mrf, int x, int y, Direction dir);
void beliefPropagation(MarkovRandomField &mrf, Direction dir);
Mat do_match(Mat leftImg, Mat rightImg, int iter, int maxDisp, float lambda, float smoothParam);
energy_t calculateMaxPosteriorProbability(MarkovRandomField &mrf);

data_cost_t calculateDataCost(cv::Mat &leftPaddingImg, cv::Mat &rightPaddingImg, const int h, const int w, const int d, const int disp);
smoothness_cost_t calculateSmoothnessCost(int i, int j, float lambda, float smoothnessParam);
#endif