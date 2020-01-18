#include<string>
#include<iostream>
#include<fstream>
#include<vector>

#include<opencv2/highgui.hpp>

#include "ioh.hpp"

IOHelper::IOHelper(/* args */)
{
    //nothing
}

IOHelper::~IOHelper()
{
}

// read param from dir
void IOHelper::setUp(std::string aim0Path, std::string aim1Path, int andisp, std::string aoutdir)
{
    im0Path = aim0Path;
    im1Path = aim1Path;
    ndisp = andisp;
    outdir = aoutdir;
    datadir = im0Path.substr(0, im0Path.find_last_of('/'));
    calibPath = datadir + "/calib.txt";
    height = -1;
    width = -1;
    ndisp = -1;
}
// read param form calib.txt
int IOHelper::readCalib(cv::Mat mats[2])
{
    std::ifstream calibParam(calibPath);
    if(!calibParam){
        std::cout<< "Can't open " << calibPath <<std::endl;
        return -1;
    }
    else{
        std::string line;
        for(int l=0; l < 2; ++l){
            getline(calibParam, line);
            line.erase(0, line.find_first_of('[')+1);
            line.erase(line.find_last_of(']'));
            std::istringstream ls(line);
            std::string token;
            std::size_t sz;
            for(int row=0; row<3; row++){
                std::getline(ls, token, ';');
                std::istringstream ts(token);
                std::string element;
                for(int col=0; col<3; col++){
                    std::getline(ts, element, ' ');
                    if(element.empty()){
                        col--;
                        continue;
                    }
                    mats[l].at<float>(row, col) = std::stof(element, &sz);
                }
            }
        }
        // while(getline(calibParam, line))
        // {
        //     std::istringstream ls(line);
        //     std::string key;
        //     std::getline(ls, key, '=');
        // }
        //TODO read height width and baseline
    }
    return 1;
}
// read image from dir
void IOHelper::readImage(cv::Mat& left_img, cv::Mat& right_img)
{
    left_img = cv::imread(im0Path, 0);
    right_img = cv::imread(im1Path, 0);
}