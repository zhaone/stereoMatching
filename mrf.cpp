#include <iostream>
#include "mrf.hpp"
// initialize the Markov Random Field
// 
void initializeMarkovRandomField(MarkovRandomField &mrf, Mat leftImg, Mat rightImg,
                                 const MarkovRandomFieldParam param)
{
    mrf.height = leftImg.rows;
    mrf.width = leftImg.cols;
    mrf.param = param;

    std::cout << "height:" << mrf.height << ", width:"
              << mrf.width << ", img size:" << leftImg.size() << std::endl;
    
    int pb = mrf.param.maxDisparity + 2;
    cv::Mat leftPaddingImg, rightPaddingImg;
    cv::copyMakeBorder(leftImg, leftPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);
    cv::copyMakeBorder(rightImg, rightPaddingImg, pb, pb, pb, pb, cv::BORDER_REPLICATE);

    mrf.grid.resize(mrf.height);
    for (int h = 0; h < mrf.height; h++)
    {
        mrf.grid[h].resize(mrf.width);
        for (int w = 0; w < mrf.width; w++)
        {
            mrf.grid[h][w].leftMessage = new msg_t[mrf.param.maxDisparity];
            mrf.grid[h][w].rightMessage = new msg_t[mrf.param.maxDisparity];
            mrf.grid[h][w].upMessage = new msg_t[mrf.param.maxDisparity];
            mrf.grid[h][w].downMessage = new msg_t[mrf.param.maxDisparity];
            mrf.grid[h][w].dataMessage = new msg_t[mrf.param.maxDisparity];
            for (int idx = 0; idx < mrf.param.maxDisparity; idx++)
            {
                // following 4 mean that there are 4 neibor, then 4 message
                // this is message m_{st}
                mrf.grid[h][w].leftMessage[idx] = 1;
                mrf.grid[h][w].rightMessage[idx] = 1;
                mrf.grid[h][w].upMessage[idx] = 1;
                mrf.grid[h][w].downMessage[idx] = 1;
                // this is message m_s
                // seems m_s(x_s) = y_s[x_s]
                // just matching cost, this code use sum of abs of sub
                
                mrf.grid[h][w].dataMessage[idx] = calculateDataCost(leftPaddingImg, rightPaddingImg, h, w, idx, mrf.param.maxDisparity);
            }
        }
    }
}
// compute and update m_{st}
// this is sum but it is multiply in paper
// no problem, matches formula 2.
void sendMsg(MarkovRandomField &mrf, const int x, const int y, const Direction dir)
{
    const int disp = mrf.param.maxDisparity;
    float sum = 0;
    
    msg_t *newMsg = new msg_t[disp];

    for (int i = 0; i < disp; i++)
    {
        msg_t minVal = UINT_MAX;
        for (int j = 0; j < disp; j++)
        {
            msg_t p = calculateSmoothnessCost(i, j, mrf.param.lambda, mrf.param.smoothnessParam);
            p += mrf.grid[y][x].dataMessage[j];

            if (dir != Left)
                p += mrf.grid[y][x].leftMessage[j];
            if (dir != Right)
                p += mrf.grid[y][x].rightMessage[j];
            if (dir != Up)
                p += mrf.grid[y][x].upMessage[j];
            if (dir != Down)
                p += mrf.grid[y][x].downMessage[j];

            minVal = min(minVal, p);
        }
        newMsg[i] = minVal;
        sum += minVal;
    }

    for (int i = 0; i < disp; i++)
    {
        switch (dir)
        {
        case Left:
            mrf.grid[y][x-1].rightMessage[i] = newMsg[i]/sum;
            break;
        case Right:
            mrf.grid[y][x+1].leftMessage[i] = newMsg[i]/sum;
            break;
        case Up:
            mrf.grid[y-1][x].downMessage[i] = newMsg[i]/sum;
            break;
        case Down:
            mrf.grid[y+1][x].upMessage[i] = newMsg[i]/sum;
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
                sendMsg(mrf, x, y, dir);
            }
        }
        break;
    case Right:
        for (int y = 1; y < h; y++)
        {
            for (int x = 0; x < w - 1; x++)
            {
                sendMsg(mrf, x, y, dir);
            }
        }
        break;
    case Up:
        for (int x = 0; x < w; x++)
        {
            for (int y = 1; y < h; y++)
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
    for(int i =0; i <mrf.param.maxDisparity; i++)
    {
        cout<<mrf.grid[100][100].leftMessage[i] << ", ";
    }
    cout<<endl;
    // cout<<mrf.grid[100][100].leftMessage<<endl;
}

energy_t calculateMaxPosteriorProbability(MarkovRandomField &mrf)
{
    const int w = mrf.width;
    const int h = mrf.height;
    // this matches the formula 3. in paper, but seems omit the m_s(x_s)
    for (int i = 0; i < mrf.height; i++)
    {
        for (int j = 0; j < mrf.width; j++)
        {
            unsigned int best = UINT_MAX;
            for (int k = 0; k < mrf.param.maxDisparity; k++)
            {
                unsigned cost = 0;

                cost += mrf.grid[i][j].leftMessage[k];
                cost += mrf.grid[i][j].rightMessage[k];
                cost += mrf.grid[i][j].upMessage[k];
                cost += mrf.grid[i][j].downMessage[k];
                cost += mrf.grid[i][j].dataMessage[k];

                if (cost < best)
                {
                    best = cost;
                    mrf.grid[i][j].bestAssignmentIndex = k;
                }
            }
        }
    }

    energy_t energy = 0;
    for (int y = 1; y < h - 1; y++)
    {
        for (int x = 1; x < w - 1; x++)
        {
            const int pos = y * mrf.width + x;
            const int bestAssignmentIndex = mrf.grid[y][x].bestAssignmentIndex;

            energy += mrf.grid[y][x].dataMessage[bestAssignmentIndex];

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y][x+1].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y][x-1].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y+1][x].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);

            energy += calculateSmoothnessCost(bestAssignmentIndex, mrf.grid[y-1][x].bestAssignmentIndex,
                                              mrf.param.lambda, mrf.param.smoothnessParam);
        }
    }

    return energy;
}

data_cost_t calculateDataCost(cv::Mat &leftPaddingImg, cv::Mat &rightPaddingImg, const int h, const int w, const int d, const int disp)
{
    int ph = disp + h;
    int pw = disp + w;
    int radius = 2;
    int dim = 2 * radius + 1;
    // Rect
    cv::Mat w1 = leftPaddingImg(cv::Rect(pw, ph, dim, dim));
    cv::Mat w2 = rightPaddingImg(cv::Rect(pw + d, ph, dim, dim));

    cv::Mat absDiff;
    cv::absdiff(w1, w2, absDiff);

    
    // return (1-ep) * std::exp(-cost/dd) + ep;
    return float(cv::sum(absDiff.mul(absDiff))[0]);
}

smoothness_cost_t calculateSmoothnessCost(const int i, const int j, const float lambda, const float smoothnessParam)
{
    const int d = i - j;
    return lambda * (abs(d)<smoothnessParam?abs(d):smoothnessParam);
}

Mat do_match(Mat leftImg, Mat rightImg, int iter, int maxDisp, float lambda, float smoothParam)
{
    MarkovRandomField mrf;
    MarkovRandomFieldParam param;

    param.iteration = iter;
    param.lambda = lambda;
    param.maxDisparity = maxDisp;
    param.smoothnessParam = smoothParam;

    initializeMarkovRandomField(mrf, leftImg, rightImg, param);
    cout<< "initiate finish" << endl;

    Mat output = Mat::zeros(mrf.height, mrf.width, CV_8U);

    for (int k = 0; k < mrf.param.iteration; k++)
    {
        // send message to left node
        beliefPropagation(mrf, Left);
        cout << "left" << endl;
        beliefPropagation(mrf, Right);
        cout << "right" << endl;
        beliefPropagation(mrf, Up);
        cout << "up" << endl;
        beliefPropagation(mrf, Down);
        cout << "down" << endl;

        const energy_t energy = calculateMaxPosteriorProbability(mrf);

        cout << "Iteration: " << k << ";  Energy: " << energy << "." << endl;

        // TODO, rectify the boder
        for (int i = 0; i < mrf.height; i++)
        {
            for (int j = 0; j < mrf.width; j++)
            {
                output.at<uchar>(i, j) = mrf.grid[i][j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
            }
        }
        cv::imshow("debug", output);
        cv::waitKey(0);
    }

    // TODO, rectify the boder
    for (int i = 0; i < mrf.height; i++)
    {
        for (int j = 0; j < mrf.width; j++)
        {
            output.at<uchar>(i, j) = mrf.grid[i][j].bestAssignmentIndex * (256 / mrf.param.maxDisparity);
        }
    }
    return output;
}