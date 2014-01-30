#include "ImageTransform1.h"

ImageTransformation1::ImageTransform1(){
	wholeFrameTransform = (Transformation){0, 0, 0, 0, 0, 1, 0};	//cos term is 1
	absoluteWholeFrameTransform = (AbsoluteTransformation){wholeFrameTransform, 0, 0};
}

ImageTransformation1::ImageTransform1(Mat img1, Mat img2){
	wholeFrameTransform = getWholeFrameTransform(img1, img2);
	frameBound = (imgBound){0, img1.cols, 0, img1.rows};
}

void ImageTransformation1::CreateAbsoluteTransform(ImageTransform1 prevTransform){
	AbsoluteTransformation absoluteWholeFrameTransform;
	float ix = prevTransform.absoluteWholeFrameTransform.idx * TRANSLATION_DECAY + wholeFrameTransform.ux1 - wholeFrameTransform.ux2;
	float iy = prevTransform.absoluteWholeFrameTransform.idy * TRANSLATION_DECAY + wholeFrameTransform.uy1 - wholeFrameTransform.uy2;
	
	absoluteWholeFrameTransform.trans = wholeFrameTransform;
	absoluteWholeFrameTransform.idx = ix;
	absoluteWholeFrameTransform.idy = iy;
	absoluteWholeFrameTransform.trans.rotation = prevTransform.absoluteWholeFrameTransform.trans.rotation * ROTATION_DECAY + wholeFrameTransform.rotation;
	absoluteWholeFrameTransform.trans.cos = cos(absoluteWholeFrameTransform.trans.rotation);
	absoluteWholeFrameTransform.trans.sin = sin(absoluteWholeFrameTransform.trans.rotation);
}

Mat ImageTransformation1::TransformImage(Mat input){
	Mat out = Mat(frameBound.maxY - frameBound.minY, frameBound.maxX - frameBound.minX , input.type());
	
	for(int y=frameBound.minY;y<frameBound.maxY;y++)
	{
		for(int x=frameBound.minX;x<frameBound.maxX;x++)
		{
			float ix, iy;
			UnTransformPointAbs(absoluteWholeFrameTransform, x, y, ix, iy);

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

			int baseIndex = out.step[0]*(y-frameBound.minY) + out.step[1]* (x-frameBound.minX);

			int newBaseIndex00 = input.step[0]*(iy0) + input.step[1]*(ix0);
			int newBaseIndex01 = input.step[0]*(iy0) + input.step[1]*(ix1);
			int newBaseIndex10 = input.step[0]*(iy1) + input.step[1]*(ix0);
			int newBaseIndex11 = input.step[0]*(iy1) + input.step[1]*(ix1);

			for(int c = 0; c<out.step[1];c++){

				float color = (1-wy) * (1-wx) * (float)input.data[newBaseIndex00 + 0]
					+ (1-wy) * (wx) * (float)input.data[newBaseIndex01 + 0]
					+ (wy) * (1-wx) * (float)input.data[newBaseIndex10 + 0]
					+ (wy) * (wx) * (float)input.data[newBaseIndex11 + 0];

				out.data[baseIndex+c] = (uchar)color;
			}
		}
	}

	return out;
}
		
ImageTransformation1::Transformation getWholeFrameTransform(Mat img1, Mat img2){
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

	//Transformation trans = densityWeightedSvd(fi1.features, fi2.features, features_found.size());
	Transformation trans = nonWeightedSvd(fi1.features, fi2.features, features_found.size());
	return trans;
}