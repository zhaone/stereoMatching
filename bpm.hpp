#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class BP
{
public:
    BP(Mat &leftImg, Mat &rightImg, const int disp, const float lambda, const float sp, int iter);
    ~BP();
    float calculateDataCost(const int h, const int w, const int d);
    void beliefPropagate(bool visualize);
    Mat maxProduct(vector<vector<Mat>> &msgCopy, int h, int w, int dir);
    Mat getDispMap();
    Mat do_match();
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
};