#ifndef JELLO_TRANSFORM_2
#define JELLO_TRANSFORM_2

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

#define     NUM_PARAMS      24

class JelloTransform2 : public ITransform {
    public:
        float ** shiftsX;
        float ** shiftsY;
        float * params;

        JelloTransform2();

        JelloTransform2(Mat img1, Mat img2, int index0, int index1);

        void CreateAbsoluteTransform(JelloTransform2 prevTransform);

        void TransformPoint(float x, float y, float &x2, float &y2);

        void TransformPointAbs(float x, float y, float &x2, float &y2);

    protected:
        vector<Transformation> jelloTransforms;     

        void CalcJelloTransform(Mat img1, Mat img2);

        void AllocateShiftMem();
};


#endif