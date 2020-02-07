#ifndef MBP_H
#define MBP_H
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class MBP
{
    private:
        // not safe
        int height;
        int width;
        int disp;
        int iter;
        Mat leftPaddingImg;
        Mat rightPaddingImg;
        float costLambda;
        Mat smoothCostMat;
        vector<vector<Mat>> msg;
        vector<vector<Mat>> obs;

    public:
        MBP(Mat &leftImg, Mat &rightImg, const int disp, const float lambda, const float sp, int iter);
        ~MBP();
        float calculateDataCost(const int h, const int w, const int d);
        void calculateDataCostThread(int sh, int eh);
        void beliefPropagate(bool visualize);
        void beliefPropagateThread(vector<vector<Mat>> &msgCopy, int sh, int eh);
        Mat maxProduct(vector<vector<Mat>> &msgCopy, int h, int w, int dir);
        Mat getDispMap();
        Mat do_match();
};
#endif