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
#include "FullFrameTransform.h"
#include "nullTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"


FullFrameTransform::FullFrameTransform(){
	wholeFrameTransform = (Transformation){0, 0, 0, 0, 0, 1, 0};	//cos term is 1
	absoluteWholeFrameTransform = (AbsoluteTransformation){wholeFrameTransform, 0, 0};
}

FullFrameTransform::FullFrameTransform(Mat img1, Mat img2, int index0, int index1, bool evalShifts){
	getWholeFrameTransform(img1, img2);
	absoluteWholeFrameTransform = (AbsoluteTransformation){wholeFrameTransform, 0, 0};
	frameBound = (imgBound){0, img1.cols, 0, img1.rows};

	#ifdef SHFITS_FILENAME
		if(evalShifts){
			evalTransforms(index0, index1, (char*)SHFITS_FILENAME);
		}
	#endif
}

void FullFrameTransform::getWholeFrameTransform(Mat img1, Mat img2){
	FeaturesInfo fi1 = extractFeaturesToTrack(img1);
	FeaturesInfo fi2 = extractFeaturesToTrack(img2);

	int length = max(fi1.features.size(), fi2.features.size());
	
	std::vector<uchar> features_found; 
	features_found.reserve(length);
	std::vector<float> feature_errors; 
	feature_errors.reserve(length);

	calcOpticalFlowPyrLK( fi1.pyramid, fi2.pyramid, fi1.features, fi2.features, features_found, feature_errors ,
		Size( WIN_SIZE, WIN_SIZE ), 5,
		 cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01 ), 0 );

	//wholeFrameTransform = densityWeightedSvd(fi1.features, fi2.features, features_found.size());
	//wholeFrameTransform = prunedNonWeightedSvd(fi1.features, fi2.features, features_found.size());
	wholeFrameTransform = WelschFit(fi1.features, fi2.features, features_found.size());
	//wholeFrameTransform = RansacNonWeightedSvd(fi1.features, fi2.features, features_found.size());
	//wholeFrameTransform = nonWeightedSvd(fi1.features, fi2.features, features_found.size());
}

void FullFrameTransform::CreateAbsoluteTransform(FullFrameTransform prevTransform){
	float ix = prevTransform.absoluteWholeFrameTransform.idx * TRANSLATION_DECAY - prevTransform.wholeFrameTransform.ux1 + prevTransform.wholeFrameTransform.ux2;
	float iy = prevTransform.absoluteWholeFrameTransform.idy * TRANSLATION_DECAY - prevTransform.wholeFrameTransform.uy1 + prevTransform.wholeFrameTransform.uy2;
	
	absoluteWholeFrameTransform.trans = wholeFrameTransform;
	absoluteWholeFrameTransform.idx = ix;
	absoluteWholeFrameTransform.idy = iy;
	absoluteWholeFrameTransform.trans.rotation = prevTransform.absoluteWholeFrameTransform.trans.rotation * ROTATION_DECAY + wholeFrameTransform.rotation;
	absoluteWholeFrameTransform.trans.cos = cos(absoluteWholeFrameTransform.trans.rotation);
	absoluteWholeFrameTransform.trans.sin = sin(absoluteWholeFrameTransform.trans.rotation);
}

void FullFrameTransform::TransformPoint(float x, float y, float &x2, float &y2){
   GenericTransformPoint(wholeFrameTransform, x, y, x2, y2);
}

void FullFrameTransform::TransformPointAbs(float x, float y, float &x2, float &y2){
	GenericTransformPointAbs(absoluteWholeFrameTransform, x, y, x2, y2);
}
