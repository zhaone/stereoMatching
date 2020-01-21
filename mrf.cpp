#include <iostream>
#include "mrf.hpp"

void initializeMarkovRandomField(MarkovRandomField &mrf, Mat leftImg, Mat rightImg,
                                 const MarkovRandomFieldParam param)
{
    // Mat leftImg = imread(leftImgPath, 0);
    // Mat rightImg = imread(rightImgPath, 0);

    // mrf.height = leftImg.cols;
    // mrf.width = leftImg.rows;
    mrf.height = leftImg.rows;
    mrf.width = leftImg.cols;
    mrf.grid.resize(mrf.height * mrf.width);
    mrf.param = param;

    for (int pos = 0; pos < mrf.height * mrf.width; pos++)
    {
        mrf.grid[pos].leftMessage = new msg_t[mrf.param.maxDisparity];
        mrf.grid[pos].rightMessage = new msg_t[mrf.param.maxDisparity];
        mrf.grid[pos].upMessage = new msg_t[mrf.param.maxDisparity];
        mrf.grid[pos].downMessage = new msg_t[mrf.param.maxDisparity];
        mrf.grid[pos].dataMessage = new msg_t[mrf.param.maxDisparity];

        for (int idx = 0; idx < mrf.param.maxDisparity; idx++)
        {
            mrf.grid[pos].leftMessage[idx] = 0;
            mrf.grid[pos].rightMessage[idx] = 0;
            mrf.grid[pos].upMessage[idx] = 0;
            mrf.grid[pos].downMessage[idx] = 0;
            mrf.grid[pos].dataMessage[idx] = 0;
        }
    }

    const int border = mrf.param.maxDisparity;

    for (int y = border; y < mrf.height - border; y++)
    {
        for (int x = border; x < mrf.width - border; x++)
        {
            for (int i = 0; i < mrf.param.maxDisparity; i++)
            {
                mrf.grid[y * mrf.width + x].dataMessage[i] = calculateDataCost(leftImg, rightImg, x, y, i);
            }
        }
    }
}

void sendMsg(MarkovRandomField &mrf, const int x, const int y, const Direction dir)
{
    const int disp = mrf.param.maxDisparity;
    const int w = mrf.width;

    msg_t *newMsg = new msg_t[disp];

    for (int i = 0; i < disp; i++)
    {
        msg_t minVal = UINT_MAX;
        for (int j = 0; j < disp; j++)
        {
            msg_t p = calculateSmoothnessCost(i, j, mrf.param.lambda, mrf.param.smoothnessParam);
            p += mrf.grid[y * w + x].dataMessage[j];

            if (dir != Left)
                p += mrf.grid[y * w + x].leftMessage[j];
            if (dir != Right)
                p += mrf.grid[y * w + x].rightMessage[j];
            if (dir != Up)
                p += mrf.grid[y * w + x].upMessage[j];
            if (dir != Down)
                p += mrf.grid[y * w + x].downMessage[j];

            minVal = min(minVal, p);
        }
        newMsg[i] = minVal;
    }

    for (int i = 0; i < disp; i++)
    {
        switch (dir)
        {
        case Left:
            mrf.grid[y * w + x - 1].leftMessage[i] = newMsg[i];
            break;
        case Right:
            mrf.grid[y * w + x - 1].rightMessage[i] = newMsg[i];
            break;
        case Up:
            mrf.grid[(y - 1) * w + x].upMessage[i] = newMsg[i];
            break;
        case Down:
            mrf.grid[(y + 1) * w + x].downMessage[i] = newMsg[i];
            break;
        default:
            break;
        }
    }
}

void beliefPropagation(MarkovRandomField &mrf, const Direction dir)
{
    const int w = mrf.width;
    const int h = mrf.height;

    switch (dir)
    {
    case Left:
        for (int y = 0; y < h; y++)
        {
            for (int x = 1; x < w; x++)
            {
                sendMsg(mrf, x, y, Left);
            }
        }
        break;
    case Right:
        for (int y = 0; y < h; y++)
        {
            for (int x = w - 1; x > 0; x--)
            {
                sendMsg(mrf, x, y, Left);
            }
        }
        break;
    case Up:
        for (int x = 0; x < w; x++)
        {
            for (int y = h - 1; y > 0; y--)
            {
                sendMsg(mrf, x, y, dir);
            }
        }
        break;
    case Down:
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h - 1; y++)
            {
                sendMsg(mrf, x, y, dir);
            }
        }
        break;
    default:
        break;
    }
}

energy_t calculateMaxPosteriorProbability(MarkovRandomField &mrf)
{
    for (int i = 0; i < mrf.grid.size(); i++)
    {
        unsigned int best = UINT_MAX;
        for (int j = 0; j < mrf.param.maxDisparity; j++)
        {
            unsigned cost = 0;

            cost += mrf.grid[i].leftMessage[j];
            cost += mrf.grid[i].rightMessage[j];
            cost += mrf.grid[i].upMessage[j];
            cost += mrf.grid[i].downMessage[j];

            if (cost < best)
            {
                best = cost;
                mrf.grid[i].bestAssignmentIndex = j;
            }
        }
    }

    const int w = mrf.width;
    const int h = mrf.height;

    energy_t energy = 0;
    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {
            const int pos = y * mrf.width + x;
            const int bestAssignmentIndex = mrf.grid[y * mrf.width + x].bestAssignmentIndex;

            energy += mrf.grid[pos].dataMessage[bestAssignmentIndex];

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y * mrf.width + x + 1].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y * mrf.width + x - 1].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[(y + 1) * mrf.width + x].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[(y - 1) * mrf.width + x].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);
        }
    }

    return energy;
}

data_cost_t calculateDataCost(Mat &leftImg, Mat &rightImg, const int x, const int y, const int disparity)
{
    const int radius = 2;

    int sum = 0;

    for (int dy = -radius; dy <= radius; dy++)
    {
        for (int dx = -radius; dx <= radius; dx++)
        {
            const int l = leftImg.at<uchar>(y + dy, x + dx);
            const int r = rightImg.at<uchar>(y + dy, x + dx - disparity);
            sum += abs(l - r);
        }
    }

    const data_cost_t avg = sum / ((radius * 2 + 1) * (radius * 2 + 1));

    return avg;
}

smoothness_cost_t calculateSmoothnessCost(const int i, const int j, const int lambda, const int smoothnessParam)
{
    const int d = i - j;
    return lambda * min(abs(d), smoothnessParam);
}

Mat do_match(Mat leftImg, Mat rightImg, int iter, int lambda, int maxDisp, int smoothParam)
{
    MarkovRandomField mrf;
    MarkovRandomFieldParam param;

    param.iteration = iter;
    param.lambda = lambda;
    param.maxDisparity = maxDisp;
    param.smoothnessParam = smoothParam;

    initializeMarkovRandomField(mrf, leftImg, rightImg, param);
    cout<< "initiate finish" << endl;

    for (int i = 0; i < mrf.param.iteration; i++)
    {
        beliefPropagation(mrf, Left);
        cout << "left" << endl;
        beliefPropagation(mrf, Right);
        cout << "right" << endl;
        beliefPropagation(mrf, Up);
        cout << "up" << endl;
        beliefPropagation(mrf, Down);
        cout << "down" << endl;

        const energy_t energy = calculateMaxPosteriorProbability(mrf);

        cout << "Iteration: " << i << ";  Energy: " << energy << "." << endl;
    }

    Mat output = Mat::zeros(mrf.height, mrf.width, CV_8U);

    for (int i = mrf.param.maxDisparity; i < mrf.height - mrf.param.maxDisparity; i++)
    {
        for (int j = mrf.param.maxDisparity; j < mrf.width - mrf.param.maxDisparity; j++)
        {
            output.at<uchar>(i, j) = mrf.grid[i * mrf.width + j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
        }
    }
    return output;
}