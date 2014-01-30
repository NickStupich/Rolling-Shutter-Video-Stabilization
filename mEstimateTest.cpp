// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>

#include "coreFuncs.h"
#include "FullFrameTransform.h"
#include "nullTransform.h"
#include "jelloTransform1.h"
#include "svd.h"
#include "mainHelpers.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"


int main(){
	vector<Point2f> corners1, corners2;

	FILE* fp = fopen("data/pts4.txt", "r");
	float x1, y1, x2, y2;
	while(fscanf(fp, "%f\t%f\t%f\t%f", &x1, &y1, &x2, &y2) != EOF){
		corners1.push_back(Point2f(x1, y1));
		corners2.push_back(Point2f(x2, y2));
	}

	printf("length of test data: %d\n", (int)corners1.size());

	printf("data[3]: %f   %f   %f   %f\n", corners1[3].x, corners1[3].y, corners2[3].x, corners2[3].y);

	Transformation t = WelschFit(corners1, corners2, (int)corners1.size());

	printf("result: %f  %f     %f   %f        %f\n", t.ux1, t.uy1, t.ux2, t.uy2, t.rotation);

}