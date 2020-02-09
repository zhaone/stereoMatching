# stereoMatching
stereo matching, disparity estimation

> Hope this project can help you or evoke your interest in stereo matching is interesting. I would be happy if you give me a star ^_^ !

This branch provides a command line version stereo matching program for depth estimation and can be evaluated on [middlebury](http://vision.middlebury.edu/stereo/eval3/). The code is tested on Ubuntu 18.04.

## result
**Sum of absolute differences. Basic.**

![Sum of absolute differences](./img/sad.png)

**Normal Cross Correlation. Not very well.**

![Normal Cross Correlation](./img/ncc.png)

**Belief Propagation. Not bad, smoother.**

![Belief Propagation](./img/mbp.png)

## build
The code depends on [Opencv 4.2](https://opencv.org/releases/), make sure you have installed lib opencv before you build this project. Libtiff4 is not avaliable on My OS(Ubuntu 18.04), so when compiling opencv lib, you'd better add `-DBUILD_TIFF=ON`. Then run:

```bash
chmod a+x ./build.sh
./build.sh
```
This will create floder `./build` and the stereo matching program is `./build/main`.

## usage
```bash
./build/main <left_img_path> <right_img_path> <max_disp> <outPath> <method> <is_visualize>
```
for example
```bash
./build/main ./img/Teddy/im0.png ./img/Teddy/im1.png 64 ./img/result.png SAD true
```
- left_img_path: path of left image
- right_img_path: path of right image
- max_disp: a conservative bound on the number of disparity levels
- outPath: path of output disparity map
- method: stereo matching mothed.
- is_visualize: whether to visualize result, `true`: visualize result

This code implements 4 methods, includes
- SAD(Sum of absolute differences)
- NCC(Normal Cross Correlation), the result is not very well
- BP(Belief Propagation), the result is smoother than SAD, but very slow.
- MBP(a multiple thread implement of Belief Propagation), faster than BP.

So you can choose from the 4 method name: `SAD, NCC, BP, MBP`. 

In order to satisfy middlebury dataset, output disparity map is 16 bit (short int) depth image in which each pixel represent the disparity of left image. If `is_visualize` is not true, you may see the output disparity map is totoally black, this is because in most case the  max_disp is much smaller than 2^16. If you want to visualize the output image in different color, you can install [cvkit](http://vision.middlebury.edu/stereo/code/cvkit/cvkit-1.7.0-src.tgz) and run
```bash
imgcmd $outPath -float -out $visualPath
sv $visualPath
```
to convert output image into float and visulize the result in other color.

## Middlebury Eval

The `run` script enables this code be evaluated by middlebury SDK. You need
1. download SDK and dataset from [Middlebury](http://vision.middlebury.edu/stereo/submit3/) 
2. install [cvkit](http://vision.middlebury.edu/stereo/code/cvkit/cvkit-1.7.0-src.tgz).
3. run:
```bash
mkdir middlebury_SDK_path/alg-XXX
cp -r . middlebury_SDK_path/alg-XXX
```
XXX is whatever algorithm name you like. 

For more middlebury infomation and eval procedure, please refer to middlebury website and the SDK REAMDE.txt.