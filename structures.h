#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace cv;

typedef struct{
	vector<Point2f> features;
	vector<Mat> pyramid;
} FeaturesInfo;

typedef struct{
    float ux1;
    float uy1;
    float ux2;
    float uy2;
    float rotation;
    float cos;
    float sin;
} Transformation;
/*
class Transformation{
public:

    float ux1;
    float uy1;
    float ux2;
    float uy2;
    float rotation;
    float cos;
    float sin;

    Transformation(float _ux1, float _uy1, float _ux2, float _uy2, float _rotation, float _cos, float _sin){
        ux1 = _ux1;
        uy1 = _uy1;
        ux2 = _ux2;
        uy2 = _uy2;
        rotation = _rotation;
        cos = _cos;
        sin = _sin;
    }

    void TransformPoint(float x, float y, float &x2, float &y2){
        x2 = (x-ux1) * cos - (y-uy1) * sin + ux2;
        y2 = (x-ux1) * sin + (y-uy1) * cos + uy2;
    }

};

class AbsoluteTransformation{
public:
    Transformation trans;
    float idx;
    float idy;

    AbsoluteTransformation(Transformation _trans, AbsoluteTransformation prevTrans){

    }

    void TransformPoint(float x, float y, float &x2, float &y2){
        trans.TransformPoint(x, y, x2, y2);
        x2 += idx;
        y2 += idy;
    }
};*/

typedef struct{
    Transformation trans;
    float idx;
    float idy;
} AbsoluteTransformation;

typedef struct{
	int minX;
	int maxX;
	int minY;
	int maxY;
} imgBound;


typedef struct{
    float x1;
    float y1;
    float x2;
    float y2;
} PointShift;

#endif