
// Video Image PSNR and SSIM
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>

#include "settings.h"
#include "structures.h"
#include "coreFuncs.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

void GenericTransformPoint(Transformation trans, float x, float y, float &x2, float &y2){
    x2 = (x-trans.ux1) * trans.cos - (y-trans.uy1) * trans.sin + trans.ux2;
    y2 = (x-trans.ux1) * trans.sin + (y-trans.uy1) * trans.cos + trans.uy2;
}

void GenericTransformPointAbs(AbsoluteTransformation absTrans, float x, float y, float &x2, float &y2){
    GenericTransformPoint(absTrans.trans, x, y, x2, y2);
    x2 += absTrans.idx;
    y2 += absTrans.idy;
}

vector<Point2f> extractCornersToTrackColor(Mat img){

	// Create Matrices (make sure there is an image in input!)
	Mat channel[3];

	// The actual splitting.
	split(img, channel);

	/*
	// Create Windows
	namedWindow("Red",1);
	namedWindow("Green",1);
	namedWindow("Blue",1);

	// Display
	imshow("Red", channel[0]);
	imshow("Green", channel[1]);
	imshow("Blue", channel[2]);
	waitKey(0);     
	*/

	vector<Point2f> result = extractCornersToTrack(channel[0], NUM_CORNERS/3);
	vector<Point2f> addition = extractCornersToTrack(channel[1], NUM_CORNERS/3);
	vector<Point2f> addition2 = extractCornersToTrack(channel[2], NUM_CORNERS/3);

	/*
	result.insert(result.end(), addition.begin(), addition.end());
	result.insert(result.end(), addition2.begin(), addition2.end());
	*/

	int startLength = (int)result.size();
	for(int i=0;i<addition.size();i++){
		bool usePoint = true;
		for(int j=0;j<startLength;j++){
			if(norm(addition[i] - result[j]) < 3.0){
				usePoint = false;
				break;
			}

		}

		if(usePoint)
			result.push_back(addition[i]);
	}

	startLength = (int)result.size();
	for(int i=0;i<addition2.size();i++){
		bool usePoint = true;
		for(int j=0;j<startLength;j++){
			if(norm(addition2[i] - result[j]) < 3.0){
				usePoint = false;
				break;
			}

		}

		if(usePoint)
			result.push_back(addition2[i]);
	}

	//printf("remaining points: %d\n", (int)result.size());

	return result;
}

#define qualityLevel 0.02
#define minDistance 5.0


vector<Point2f> extractCornersRecursive(Mat img){
	return extractCornersRecursiveInner(img, NUM_CORNERS, Point2f(0, 0));
}

int *finalStageCounts;

vector<Point2f> extractCornersRecursiveInner(Mat img, int numCorners, Point2f offset){
	vector<Point2f> result;

	goodFeaturesToTrack( img, result,numCorners,qualityLevel,minDistance,cv::Mat());

	int counts[4];
	memset(counts, 0, 4*sizeof(int));

	int halfHeight = img.rows/2;
	int halfWidth = img.cols/2;
	int minCount = result.size() / 10;	//min of 10% in each quarter

	for(int i=0;i<result.size();i++){
		int index = 0;
		if(result[i].y > halfHeight)
			index++;
		if(result[i].x > halfWidth)
			index+=2;

		counts[index]++;
	}

	//printf("total counts: %d      counts: %d %d %d %d\n", numCorners, counts[0], counts[1], counts[2], counts[3]);
	bool countsUneven = false;
	if(counts[0] < minCount)	countsUneven = true;
	else if(counts[1] < minCount) 	countsUneven = true;
	else if(counts[2] < minCount) 	countsUneven = true;
	else if(counts[3] < minCount) 	countsUneven = true;

	if(countsUneven && numCorners > 4){
		result.erase(result.begin(), result.end());

		//printf("counts are too uneven, doing another level\n");

		Mat topHalf = img.rowRange(0, halfHeight);
		Mat topLeft = topHalf.colRange(0, halfWidth);
		Mat topRight = topHalf.colRange(halfWidth, img.cols);
		Point2f topLeftOffset(0, 0);
		Point2f topRightOffset(halfWidth, 0);

		Mat bottomHalf = img.rowRange(halfHeight, img.rows);
		Mat bottomLeft = bottomHalf.colRange(0, halfWidth);
		Mat bottomRight = bottomHalf.colRange(halfWidth, img.cols);
		Point2f bottomLeftOffset(0, halfHeight);
		Point2f bottomRightOffset(halfWidth, halfHeight);

		vector<Point2f> q0 = extractCornersRecursiveInner(topLeft, numCorners/4, topLeftOffset);
		vector<Point2f> q1 = extractCornersRecursiveInner(topRight, numCorners/4, topRightOffset);
		vector<Point2f> q2 = extractCornersRecursiveInner(bottomLeft, numCorners/4, bottomLeftOffset);
		vector<Point2f> q3 = extractCornersRecursiveInner(bottomRight, numCorners/4, bottomRightOffset);

		result.insert(result.end(), q0.begin(), q0.end());
		result.insert(result.end(), q1.begin(), q1.end());
		result.insert(result.end(), q2.begin(), q2.end());
		result.insert(result.end(), q3.begin(), q3.end());
			
	} else{
		int depth = (int)log2(NUM_CORNERS / numCorners)/2;
		finalStageCounts[depth] ++;

		 if(result.size() > 0){

			#if DO_CORNER_SUBPIX == 1
				cornerSubPix( img, result, Size( WIN_SIZE, WIN_SIZE ), Size( -1, -1 ), 
							  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
			#endif
		}
	}

	for(int i=0;i<result.size();i++){
		result[i] += offset;
	}

	return result;
}

vector<Point2f> extractCornersToTrack(Mat img){
	return extractCornersToTrack(img, NUM_CORNERS);
}

vector<Point2f> extractCornersToTrack(Mat img, int numCorners){
	vector<Point2f> corners;
	
	//double qualityLevel = 0.02; 
	//double minDistance = 5.0;
	
	int type = 2;
	
	switch(type){
	case 0: goodFeaturesToTrack( img,corners,numCorners,qualityLevel,minDistance,cv::Mat());
		break;
	case 1: goodFeaturesToTrack( img,corners,numCorners,qualityLevel,minDistance,cv::Mat(), 3, 1); //harris detector
		break;
	case 2:
		int numCols = 15;
		int numRows = numCols;
		for(int col = 0;col<numCols;col++)
		{
			for(int row = 0;row<numRows;row++)
			{
				int xLow = img.cols * col / numCols;
				int xHigh = img.cols * (col+1) / numCols;
				int yLow = img.rows * row / numRows;
				int yHigh = img.rows * (row+1) / numRows;
				
				Mat m1 = img.rowRange(yLow, yHigh);
				Mat m = m1.colRange(xLow, xHigh);
				
				Point2f offset(xLow, yLow);
				vector<cv::Point2f> segmentCorners;
				goodFeaturesToTrack( m,segmentCorners,numCorners/(numCols * numRows),qualityLevel,minDistance,cv::Mat());
				for(int i=0;i<(int)segmentCorners.size();i++)
				{
					corners.push_back(segmentCorners[i] + offset);
				}
			}
		}
		break;
		
	}
	
	#if DO_CORNER_SUBPIX == 1
	//printf("size of input array: %d\n", (int)corners.size());
	//corners.reserve(numCorners);
	
	cornerSubPix( img, corners, Size( WIN_SIZE, WIN_SIZE ), Size( -1, -1 ), 
				  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );
	#endif
	
	if(SHOW_CORNERS)
        {
            for(int i=0;i<(int)corners.size();i++)
            {
                    circle(img, corners[i], 2, Scalar(255, 0, 0));	
            }
            imshow("Window1", img); 
            cvWaitKey(10000);
            
        }
	return corners;
}

FeaturesInfo extractFeaturesToTrack(Mat img){
	vector<Point2f> corners = extractCornersToTrack(img);

	vector<Mat> pyramid;
	buildOpticalFlowPyramid(img, pyramid, Size(WIN_SIZE, WIN_SIZE), 3);
	
	FeaturesInfo fi;
	fi.features = corners;
	fi.pyramid = pyramid;
	
	return fi;
}

vector<Mat> getAllInputFrames(CvCapture* capture, int numFrames){
    vector<Mat> result;
    
    cvSetCaptureProperty(capture,CV_CAP_PROP_POS_FRAMES,0);
	
    for(int i=0;i<numFrames;i++)
    {
		Mat m(cvCloneImage(cvQueryFrame(capture)));
		result.push_back(m);
    }
    
    return result;
}

Mat matToGrayscale(Mat m){
	Mat greyMat;
	cvtColor(m, greyMat, CV_BGR2GRAY);
	return greyMat;
}

vector<Mat> convertFramesToGrayscale(vector<Mat> input){
	vector<Mat> result;
	for(int i=0;i<(int)input.size();i++)
	{
		result.push_back(matToGrayscale(input[i]));
	}
	return result;
}

void writeVideo(vector<Mat> frames, int fps, string filename){
	int codec = CV_FOURCC('M', 'J', 'P', 'G');

	int width = frames[0].cols;
	int height = frames[0].rows;
	
	VideoWriter outputVideo;
	
	#ifdef ROTATE90
	Size size(height, width);
	#else
	Size size(width, height);
	#endif

	outputVideo.open(filename, codec, fps, size, true);
	if(!outputVideo.isOpened()){
		printf("output video failed to open\n");
		exit(1);
	}

	cvNamedWindow("window", CV_WINDOW_NORMAL );

	for(int i=0;i<(int)frames.size();i++){

		printf("\b\b\b\b\b\b\b\b\b\b\b\b\b%d/%d", i, (int) frames.size());
		fflush(stdout);
		
		Mat frame = frames[i];

		#ifdef ROTATE90
		outputVideo.write(frame.t());
		#else
		outputVideo.write(frame);
		#endif
	}
}

int GetPointsToTrack(Mat img1, Mat img2, vector<Point2f> &corners1, vector<Point2f> &corners2){

	Size img_sz = img1.size();
	Mat imgC(img_sz,1);
 
	int win_size = 15;
	int maxCorners = NUM_CORNERS;

	corners1 = extractCornersToTrack(img1);
	//corners1 = extractCornersRecursive(img1);

	corners1.reserve(maxCorners); 
	corners2.reserve(maxCorners);

	CvSize pyr_sz = Size( img_sz.width+8, img_sz.height/3 );
	
	std::vector<uchar> features_found; 
	features_found.reserve(maxCorners);
	std::vector<float> feature_errors; 
	feature_errors.reserve(maxCorners);
    
	calcOpticalFlowPyrLK( img1, img2, corners1, corners2, features_found, feature_errors ,
		Size( win_size, win_size ), 5,
		 cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );

	return (int) features_found.size();
}
