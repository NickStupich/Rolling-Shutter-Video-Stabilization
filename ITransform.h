#ifndef ITRANSFORM_H
#define ITRANSFORM_H

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <numeric>

#include "svd.h"
#include "structures.h"
#include "settings.h"
#include "coreFuncs.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

class ITransform
{
	public:
		static imgBound frameBound;
		static vector<float> frameErrors;

		static int imgHeight, imgWidth;
		static int processedFrameCount;

		ITransform();
		ITransform(Mat img1, Mat img2, int index0, int index1);
		Mat TransformImage(Mat input);
		virtual void TransformPoint(float x, float y, float &x2, float &y2) = 0;
		virtual void TransformPointAbs(float x, float y, float &x2, float &y2) = 0;
		static void analyzeTransformAccuracies();

	protected:
		void evalTransforms(int indx0, int index1, char* baseShiftFilename);
};

#endif