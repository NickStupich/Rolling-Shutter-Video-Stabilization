#ifndef SVD_H
#define SVD_H

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "structures.h"

#include <armadillo>

using namespace std;
using namespace cv;

Transformation prunedWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights);

Transformation prunedNonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length);

Transformation weightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights);

Transformation nonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length);

vector<float> getDensityWeights(vector<float> dxs, vector<float> dys, int length);

Transformation densityWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length);

Transformation RansacNonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length);

#define	START_W	1.0e4
#define	END_W		4.0
#define	NUM_STEPS	20

#define	NEWTON_STABILITY_LIMIT	5
#define	NEWTON_STABILITY_LIMIT_ROTATION	(2.0 * 3.14 / 180)


Transformation WelschFit(vector<Point2f> corners1, vector<Point2f> corners2, int length);

Transformation WelschFitWeighted(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights);

#endif