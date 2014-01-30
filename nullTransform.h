#ifndef NULL_TRANSFORM_H
#define NULL_TRANSFORM_H

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
#include "ITransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

class NullTransform : public ITransform{
	public:	 
		NullTransform();

		NullTransform(Mat img1, Mat img2, int index0, int index1);
		
		void CreateAbsoluteTransform(NullTransform prevTransform);
		
		//Mat TransformImage(Mat input);

		void TransformPoint(float x, float y, float &x2, float &y2);

		void TransformPointAbs(float x, float y, float &x2, float &y2);
};

vector<PointShift> loadActualShifts(char* filename);

#endif