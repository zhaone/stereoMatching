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

typedef unsigned int msg_t;
typedef unsigned int energy_t;
typedef unsigned int smoothness_cost_t;
typedef unsigned int data_cost_t;

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
    int maxDisparity, lambda, iteration, smoothnessParam;
};

struct MarkovRandomField
{
    vector<MarkovRandomFieldNode> grid;
    MarkovRandomFieldParam param;
    int height, width;
};

void initializeMarkovRandomField(MarkovRandomField &mrf, Mat leftImg, Mat rightImg, MarkovRandomFieldParam param);
void sendMsg(MarkovRandomField &mrf, int x, int y, Direction dir);
void beliefPropagation(MarkovRandomField &mrf, Direction dir);
Mat do_match(Mat leftImg, Mat rightImg, int iter, int lambda, int maxDisp, int smoothParam);
energy_t calculateMaxPosteriorProbability(MarkovRandomField &mrf);

data_cost_t calculateDataCost(Mat &leftImg, Mat &rightImg, int x, int y, int disparity);
smoothness_cost_t calculateSmoothnessCost(int i, int j, int lambda, int smoothnessParam);
#endif