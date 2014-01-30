#ifndef CORE_FUNCS_H
#define CORE_FUNCS_H

// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>

#include "settings.h"
#include "structures.h"
#include "svd.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

extern int *finalStageCounts;

void GenericTransformPoint(Transformation trans, float x, float y, float &x2, float &y2);
void GenericTransformPointAbs(AbsoluteTransformation absTrans, float x, float y, float &x2, float &y2);
vector<Point2f> extractCornersToTrack(Mat img);
vector<Point2f> extractCornersToTrack(Mat img, int numCorners);
FeaturesInfo extractFeaturesToTrack(Mat img);
vector<Mat> getAllInputFrames(CvCapture* capture, int numFrames);
Mat matToGrayscale(Mat m);
vector<Mat> convertFramesToGrayscale(vector<Mat> input);
void writeVideo(vector<Mat> frames, int fps, string filename);
int GetPointsToTrack(Mat img1, Mat img2, vector<Point2f> &corners1, vector<Point2f> &corners2);
vector<Point2f> extractCornersToTrackColor(Mat img);
vector<Point2f> extractCornersRecursive(Mat img);
vector<Point2f> extractCornersRecursiveInner(Mat img, int numCorners, Point2f offset);

#endif