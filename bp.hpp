#include<opencv2/opencv.hpp>
#include<vector>

using namespace cv;
using namespace std;

class BP
{
private:
    int height;
    int width;
    int disp;
    int iter;
    Mat leftImg;
    Mat rightImg;
    Mat smoothCostMat;
    vector<vector<Mat>> msg;
    vector<vector<Mat>> obs;
public:
    BP(Mat &leftImg, Mat &rightImg, const int disp, const float lambda, const float sp, int iter);
    ~BP();
    float calculateDataCost(cv::Mat &leftPaddingImg, cv::Mat &rightPaddingImg, const int h, const int w, const int d);
    void beliefPropagate();
    Mat maxProduct(vector<vector<Mat>> &msgCopy, int h, int w, int dir);
    Mat getDispMap();
    Mat do_match();
};
