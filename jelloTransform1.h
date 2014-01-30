#ifndef JELLO_TRANSFORM_1
#define JELLO_TRANSFORM_1

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
#include "FullFrameTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

class JelloTransform1 : public FullFrameTransform {
    public:
        float ** shiftsX;
        float ** shiftsY;

        int jelloMinX, jelloMinY;

        JelloTransform1();

        JelloTransform1(Mat img1, Mat img2, int index0, int index1);

        void CreateAbsoluteTransform(JelloTransform1 prevTransform);

        void TransformPoint(float x, float y, float &x2, float &y2);

        void TransformPointAbs(float x, float y, float &x2, float &y2);

    protected:
        vector<Transformation> jelloTransforms; 	

        void CalcJelloTransform(Mat img1, Mat img2);

        void GetJelloShift(float x, float y, float &x2, float &y2);

        Mat FullFrameTransformImage(Mat input);

        void FullFrameTransformPoint(float x, float y, float &x2, float &y2);

        void AllocateShiftMem();
};


#endif