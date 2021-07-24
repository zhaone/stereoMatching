# stereoMatching
stereo matching, disparity estimation. 

**A Camera Calibration repo of mine is available [here](https://github.com/zhaone/ProjectStereo).**

> Hope this project can help you or evoke your interest in stereo matching is interesting. I would be happy if you give me a star ^_^ !

Master branch provides GUI version stereo matching program for depth estimation. [cli](https://github.com/zhaone/stereoMatching/tree/cli) branch provides a command line version and can be evaluated on [middlebury](http://vision.middlebury.edu/stereo/eval3/). The code is tested on Ubuntu 18.04.

## result
**Sum of absolute differences. Basic.**

![Sum of absolute differences](./img/sad.png)

**Normal Cross Correlation. Not very well.**

![Normal Cross Correlation](./img/ncc.png)

**Belief Propagation. Not bad, smoother.**

![Belief Propagation](./img/mbp.png)

## build
The code depends on [Opencv 4.2](https://opencv.org/releases/) and [qt](https://www.qt.io/) make sure you have installed lib opencv and qt before you build this project. Libtiff4 is not avaliable on My OS(Ubuntu 18.04), so when compiling opencv lib, you'd better add `-DBUILD_TIFF=ON`. 

To build this project, just use qt-creator open seteroMatchingQt.pro and compile.

## usage
This is the GUI program. 

![gui](./img/gui.png)

To compute disparity of two image:
1. choose left and right image path. eg. leftImage: ./img/Teddy/im0.png leftImage:./img/Teddy/im1.png
2. choose stereo matching method. This code implements 4 methods, includes
   - SAD(Sum of absolute differences)
   - NCC(Normal Cross Correlation), the result is not very well
   - Belief Propagation, the result is smoother than SAD, but very slow.
   - a multiple thread implement of Belief Propagation, faster than BP.
3. set algorithm args.
   -  Window Size: for all 4 algorithms, size of window when compute matching cost of each pixel.
   -  Max disp: for all 4 algorithms, the maximum disparity in two image.
   -  iteration: for BP and MBP, number of iteration of BP and MBP algorithms.
   -  smooth_cost: for BP and MBP. The bigger smooth_cost is, the smoother output disparity map is.
4. click `start matching` button and wait program compute and display final disparity map.

You may need wait for some time to get final disparity map for left image, when using BP or MBP algorithm and setting a big iteration number, it may cost quite a long time. 

The program will display disparity map, image height, image width, and cost time after finishing computing.
