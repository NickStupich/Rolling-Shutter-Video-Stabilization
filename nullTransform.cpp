#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <numeric>

#include "nullTransform.h"
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



NullTransform::NullTransform(){
}


NullTransform::NullTransform(Mat img1, Mat img2, int index0, int index1){
	frameBound = (imgBound){0, img1.cols, 0, img1.rows};

/*
	vector<Point2f> corners1, corners2;
	int length = GetPointsToTrack(img1, img2, corners1, corners2);
*/

	#ifdef SHFITS_FILENAME

		evalTransforms(index0, index1, (char*)SHFITS_FILENAME);

	#endif
}

void NullTransform::CreateAbsoluteTransform(NullTransform prevTransform){
}

void NullTransform::TransformPoint(float x, float y, float &x2, float &y2){
	x2 = x;
	y2 = y;
}

void NullTransform::TransformPointAbs(float x, float y, float &x2, float &y2){
	x2 = x;
	y2 = y;
}

vector<PointShift> loadActualShifts(char* filename){

	vector<PointShift> frameShifts;

	FILE* fp = fopen(filename, "r");
	float x1, y1, x2, y2;
	
	while(fscanf(fp, "%f\t%f\t%f\t%f", &x1, &y1, &x2, &y2) != EOF){

		PointShift ps = {x1, y1, x2, y2};
		frameShifts.push_back(ps);
	}
	fclose(fp);

	return frameShifts;
}
