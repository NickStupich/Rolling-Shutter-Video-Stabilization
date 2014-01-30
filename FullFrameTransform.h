#ifndef FULL_FRAME_TRANSFORM_H
#define FULL_FRAME_TRANSFORM_H

// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>

#include "nullTransform.h"
#include "svd.h"
#include "structures.h"
#include "settings.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"


class FullFrameTransform : public ITransform {
	public:
		Transformation wholeFrameTransform;
		AbsoluteTransformation absoluteWholeFrameTransform;

		FullFrameTransform();

		FullFrameTransform(Mat img1, Mat img2, int index0, int index1, bool evalShifts = true);

		void CreateAbsoluteTransform(FullFrameTransform prevTransform);
		
		void TransformPoint(float x, float y, float &x2, float &y2);

		void TransformPointAbs(float x, float y, float &x2, float &y2);

		void getWholeFrameTransform(Mat img1, Mat img2);
};


#endif