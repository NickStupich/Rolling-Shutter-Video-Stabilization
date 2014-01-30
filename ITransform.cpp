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


int ITransform::imgHeight;
int ITransform::imgWidth;

imgBound ITransform::frameBound;
vector<float> ITransform::frameErrors;
int ITransform::processedFrameCount;

ITransform::ITransform(){}
ITransform::ITransform(Mat img1, Mat img2, int index0, int index1){}

Mat ITransform::TransformImage(Mat input){
	Mat out = Mat(input.rows, input.cols, input.type());

	for(int y=frameBound.minY;y<frameBound.maxY;y++)
	{
		for(int x=frameBound.minX;x<frameBound.maxX;x++)
		{
			float ix, iy;
			TransformPointAbs(x, y, ix, iy);

			int baseIndex = out.step[0]*(y) + out.step[1]* (x);

			if(ix < 0 || ix > (input.cols-1) || iy < 0 || iy > (input.rows-1)){

				for(int c = 0; c<(int)out.step[1];c++){
					out.data[baseIndex+c] = 0;
				}
				
				if(ix < 0)
					frameBound.minX = max(x+1, frameBound.minX);
				if(ix > input.cols-1)
					frameBound.maxX = min(x-1, frameBound.maxX);
				if(iy < 0)
					frameBound.minY = max(y+1, frameBound.minY);
				if(iy > input.rows-1)
					frameBound.maxY = min(y-1, frameBound.maxY);

				//printf("frame bound now: x: %d  %d      y: %d %d\n", frameBound.minX, frameBound.maxX, frameBound.minY, frameBound.maxY);
				
			} else {
				float wx = fmod(ix, 1);
				float wy = fmod(iy, 1);

				if(wx < 0)
					wx += 1.0F;
				if(wy < 0)
					wy += 1.0F;

				int iy0 = (int) floor(iy);
				int iy1 = (int) ceil(iy);
				int ix0 = (int) floor(ix);
				int ix1 = (int) ceil(ix);

				int newBaseIndex00 = input.step[0]*(iy0) + input.step[1]*(ix0);
				int newBaseIndex01 = input.step[0]*(iy0) + input.step[1]*(ix1);
				int newBaseIndex10 = input.step[0]*(iy1) + input.step[1]*(ix0);
				int newBaseIndex11 = input.step[0]*(iy1) + input.step[1]*(ix1);

				for(int c = 0; c<(int)out.step[1];c++){
					float color = (1-wy) * (1-wx) * (float)input.data[newBaseIndex00 + c]
					+ (1-wy) * (wx) * (float)input.data[newBaseIndex01 + c]
					+ (wy) * (1-wx) * (float)input.data[newBaseIndex10 + c]
					+ (wy) * (wx) * (float)input.data[newBaseIndex11 + c];

					out.data[baseIndex+c] = (uchar)color;
				}
			}
		}
	}

	return out;
}

void ITransform::analyzeTransformAccuracies(){
	if(frameErrors.size() == 0){
		printf("no shifts were analyzed during the run\n");
		return;
	}

	float mean = std::accumulate(frameErrors.begin(), frameErrors.end(), 0.0) / (float)frameErrors.size();
	printf("mean error: %f\n", mean);
}

void ITransform::evalTransforms(int index0, int index1, char* baseShiftFilename){
	char filename[100];
	sprintf(filename, baseShiftFilename, index0);
	vector<PointShift> shifts1 = loadActualShifts(filename);

	sprintf(filename, baseShiftFilename, index1);
	vector<PointShift> shifts2 = loadActualShifts(filename);

	float meanFrameDist = 0;
	for(int i=0;i<(int)shifts1.size();i++){

		float x2 = shifts2[i].x1;
		float y2 = shifts2[i].y1;

		float dx = shifts2[i].x2 - shifts1[i].x2;
		float dy = shifts2[i].y2 - shifts1[i].y2;

		float ix, iy;
		TransformPoint(x2, y2, ix, iy);

		float ex = dx - (x2 - ix);
		float ey = dy - (y2 - iy);
		float d = (sqrt(ex*ex + ey * ey));

		meanFrameDist += d;
	}

	meanFrameDist /= (float) shifts1.size();
	printf("mean frame distance: %f\n", meanFrameDist);

	frameErrors.push_back(meanFrameDist);
}
