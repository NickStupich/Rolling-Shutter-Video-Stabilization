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

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

template <class TRANSFORM>
vector<TRANSFORM> getImageTransformsFromGrey(vector<Mat> greyInput);

template <class TRANSFORM>
vector<Mat> transformMats(vector<Mat> input, vector<TRANSFORM> transforms);

template <class TRANSFORM>
void evalTransform();