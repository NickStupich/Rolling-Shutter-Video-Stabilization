#ifndef JELLO_COMPLEX_1
#define JELLO_COMPLEX_1

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

class JelloComplex1 : public ITransform {
    public:
        float ** shiftsX;
        float ** shiftsY;
        float * params;
        static float centerX;

        int jelloMinX, jelloMinY;

        JelloComplex1();

        JelloComplex1(Mat img1, Mat img2, int index0, int index1);

        JelloComplex1(vector<Point2f> corners1, vector<Point2f> corners2, int length);

        void CreateAbsoluteTransform(JelloComplex1 prevTransform);

        void TransformPoint(float x, float y, float &x2, float &y2);

        void TransformPointAbs(float x, float y, float &x2, float &y2);

        void CalculateModelParameters(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* &params);

        float FullModelCostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float w, float* params);

        float FullModelCostLs(vector<Point2f> corners1, vector<Point2f> corners2, float* params);
    protected:
        vector<Transformation> jelloTransforms; 	

        void CalcJelloTransform(Mat img1, Mat img2);
        void GetSineEstimatesWeighted(vector<float> ys, vector<float> diffs, vector<float> weights, float* &result);
        void ImproveSineEstimates(vector<float> ys, vector<float> diffs, vector<float> weights, int length, float* params, float* &updates, float &lambda);
        float SineEstimatesCost(vector<float> ys, vector<float> diffs, vector<float> weights, float* params);

        void FullModelWelschFit(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates, float &lambda, float w);
        void AllocateShiftMem();

        void Model2NewtonIterationLS(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates);
};


#endif