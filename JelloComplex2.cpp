#define _USE_MATH_DEFINES

#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <time.h>
#include <numeric>
#include <math.h>

#include "svd.h"
#include "structures.h"
#include "settings.h"
#include "coreFuncs.h"
#include "JelloComplex2.h"
#include "nullTransform.h"

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/imgproc/imgproc_c.h"

#include <armadillo>

#define FLOAT_TYPE 		float
#define NUM_FREQUENCIES 	2

#define LAMBDA_INCREASE	2
#define LAMBDA_DECREASE	2
#define MAX_LAMBDA		1000

#define EXTRACT_PARAMS(params) 	float r0 = params[0];\
float dx0 = params[1];\
float dy0 = params[2];\
float r1 = params[3];\
float dx1 = params[4];\
float dy1 = params[5];\
float dx2_0 = params[6];\
float dx3_0 = params[7];\
float dx4_0 = params[8];\
float dx2_1 = params[9];\
float dx3_1 = params[10];\
float dx4_1 = params[11];\
float dy2_0 = params[12];\
float dy3_0 = params[13];\
float dy4_0 = params[14];\
float dy2_1 = params[15];\
float dy3_1 = params[16];\
float dy4_1 = params[17];\
float r2_0 = params[18];\
float r3_0 = params[19];\
float r4_0 = params[20];\
float r2_1 = params[21];\
float r3_1 = params[22];\
float r4_1 = params[23];

float JelloComplex2::centerX;

bool doTest = false;

#define START_LAMBDA 		0.01

#define MODEL2_START_W		1000
#define MODEL2_END_W		7
#define MODEL2_START_LAMBDA	START_LAMBDA
#define MODEL2_STEPS		20


#define FULL_MODEL_LAMBDA	START_LAMBDA
#define FULL_MODEL_STEPS		10
#define FULL_MODEL_START_W	10
#define FULL_MODEL_END_W		4

#define F_LOW 	0.002
#define F_HIGH	0.1
#define F_STEPS	20
#define SIN_MODEL_LAMBDA		START_LAMBDA
#define SIN_MODEL_STEPS		10
#define SIN_MODEL_START_W		10
#define SIN_MODEL_END_W		4
#define SINE_MODEL_WEIGHTING_W	10

vector<float> JelloComplex2::allWelschCosts;
vector<float> JelloComplex2::initialWelschCosts;
vector<float> JelloComplex2::fullFrameWelschCosts;

JelloComplex2::JelloComplex2(){
	AllocateShiftMem();
}

JelloComplex2::JelloComplex2(Mat img1, Mat img2, int index0, int index1){
	frameBound = (imgBound){0, img1.cols, 0, img1.rows};
	centerX = img1.cols / 2.0;

	CalcJelloTransform(img1, img2);

	#ifdef SHFITS_FILENAME

		evalTransforms(index0, index1, (char*)SHFITS_FILENAME);

	#endif
	
	AllocateShiftMem();

}

/* *** Test constructor ***    */
JelloComplex2::JelloComplex2(vector<Point2f> corners1, vector<Point2f> corners2, int length){
	params = new float[NUM_PARAMS];
	memset(params, 0, NUM_PARAMS*sizeof(float));
	AllocateShiftMem();
	CalculateModelParameters(corners1, corners2, length, params);
}

void JelloComplex2::AllocateShiftMem(){
	shiftsX = new float*[imgHeight];
	shiftsY = new float*[imgHeight];

	for(int row=0;row<imgHeight;row++){
		shiftsX[row] = new float[imgWidth];
		shiftsY[row] = new float[imgWidth];
		memset(shiftsX[row], 0, imgWidth * sizeof(float));
		memset(shiftsY[row], 0, imgWidth * sizeof(float));
	}
}

void JelloComplex2::CalculateModelParameters(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* &params){
	float * updatesModelLevel2 = new float[6];
	float lambda = MODEL2_START_LAMBDA;
	float startW = MODEL2_START_W;
	float endW = MODEL2_END_W;
	float steps = MODEL2_STEPS;
	memset(params, 0, NUM_PARAMS);

	float startCost = FullModelCostWelsch(corners1, corners2, 4, params);
	//printf("starting cost (welsch(4)): %f\n", startCost);
	initialWelschCosts.push_back(startCost);

	//calculate the 6 model parameters for dx, dy, full rotation, dx1*y, dy1*y, r*y
	//width is reduced but not all the way since there will still be wobble of inliers from jello
	for(int iter=0;iter<steps;iter++){
		float w = pow(10,  log10(startW) + (log10(endW) - log10(startW)) * (iter / (steps-1)));
		model2LMIterationWelsch(corners1, corners2, length, params, updatesModelLevel2, lambda, w);
		for(int i=0;i<6;i++){params[i] -= updatesModelLevel2[i];}
	}
	
	float fullFrameCost = FullModelCostWelsch(corners1, corners2, 4, params);
	//printf("full frame cost (welsch(4)): %f\n", startCost);
	fullFrameWelschCosts.push_back(fullFrameCost);
	
	
	float* roundParams = &params[6];
	

	vector<float> ys(length);
	vector<float> diffs(length);
	vector<float> weights(length);
	vector<float> startingWeights(length);
	fill(weights.begin() ,weights.end(), 1.0);

	float ex, ey, eTotal;

	//calculate starting weights (expected inlier/outlier for each point) based on how far they're off the full frame model
	float startingWeightsSum = 0;
	for(int i=0;i<length;i++){
		float x, y, x2, y2;
		x = corners1[i].x;
		y = corners1[i].y;
		x2 = corners2[i].x;
		y2 = corners2[i].y;

		EXTRACT_PARAMS(params)
		float x2Pred = dx0 + dx1*y + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
		float y2Pred = dy0 + dy1*y  + x*sin(r0 + r1*y) + y*cos(r0 + r1*y);
		
		float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
		float w = exp(-d/(SINE_MODEL_WEIGHTING_W*SINE_MODEL_WEIGHTING_W));
		startingWeights[i] = w;
		startingWeightsSum += w;
	}

	//normalize stuff so we don't need to normalize it later
	for(int i=0;i<length;i++){
		startingWeights[i] *= (float)length / startingWeightsSum;
	}



	eTotal = FullModelCostWelschXY(corners1, corners2, FULL_MODEL_END_W, params, ex, ey);
	//printf("0 ex: %f  ey: %f    total: %f\n", ex, ey, eTotal);
	for(int frequencyNum = 0; frequencyNum < NUM_FREQUENCIES; frequencyNum++){
		float xPred, yPred;
		for(int i=0;i<length;i++){
			ys[i] = corners1[i].y;
			TransformPoint(corners1[i].x, corners1[i].y, xPred, yPred);
			diffs[i] = corners2[i].x - xPred;
		}

		GetSineEstimatesWeighted2(ys, diffs, weights, startingWeights, roundParams);
		roundParams = &roundParams[3];
		eTotal = FullModelCostWelschXY(corners1, corners2, FULL_MODEL_END_W, params, ex, ey);
		//printf("%d ex: %f  ey: %f    total: %f\n", frequencyNum+1, ex, ey, eTotal);
	}
	//printf("\n\n");


	//printf("cost after 12 params fit (welsch(4)): %f\n", FullModelCostWelsch(corners1, corners2, 4, params));

	for(int frequencyNum = 0; frequencyNum < NUM_FREQUENCIES; frequencyNum++){
		float xPred, yPred;
		for(int i=0;i<length;i++){
			ys[i] = corners1[i].y;
			TransformPoint(corners1[i].x, corners1[i].y, xPred, yPred);
			diffs[i] = corners2[i].y - yPred;
		}

		GetSineEstimatesWeighted2(ys, diffs, weights, startingWeights, roundParams);

		roundParams = &roundParams[3];
		
	}


	//printf("cost after 18 params fit (welsch(4)): %f\n", FullModelCostWelsch(corners1, corners2, 4, params));

	for(int frequencyNum = 0; frequencyNum < NUM_FREQUENCIES; frequencyNum++){
		if(frequencyNum == 1)
			doTest = true;

		
		float xPred, yPred;
		for(int i=0;i<length;i++){
			ys[i] = corners1[i].y;
			TransformPoint(corners1[i].x, corners1[i].y, xPred, yPred);
			diffs[i] = corners2[i].y - yPred;
			weights[i] = (corners1[i].x - centerX) / (2*centerX);
		}

		GetSineEstimatesWeighted2(ys, diffs, weights, startingWeights, roundParams);

		roundParams = &roundParams[3];
	}

	
	//printf("params for final optimization:\n");
	//for(int i=0;i<24;i++)	printf("%f\t", params[i]);
	//printf("\n\n");

	//printf("cost before final optimization (welsch(4)): %f\n", FullModelCostWelsch(corners1, corners2, 4, params));
	

	lambda = FULL_MODEL_LAMBDA;
	startW = FULL_MODEL_START_W;
	endW = FULL_MODEL_END_W;
	steps = FULL_MODEL_STEPS;
	for(int i=0;i<steps;i++){
		float w = pow(10, (log10(endW) - log10(startW)) * i / (steps-1) + log10(startW));
		float* updates = new float[NUM_PARAMS];
		FullModelWelschFit(corners1, corners2, length, params, updates, lambda, w);
		for(int j=0;j<NUM_PARAMS;j++){
			params[j] -= updates[j];
		}
	}

	
	float finalCost = FullModelCostWelsch(corners1, corners2, 4, params);
	allWelschCosts.push_back(finalCost);
	//printf("cost after final optimization (welsch(4)): %f\n", finalCost);
}

void JelloComplex2::FullModelWelschFit(vector<Point2f> corners1, vector<Point2f> corners2, int length, FLOAT_TYPE* params, FLOAT_TYPE* &updates, FLOAT_TYPE &lambda, FLOAT_TYPE w){

	arma::Col<FLOAT_TYPE> fVector(NUM_PARAMS);
	arma::Mat<FLOAT_TYPE> jacob(NUM_PARAMS,NUM_PARAMS);
	fVector.zeros();
	jacob.zeros();

	float jacobianDiagonals[24];
	float newParams[24];

	FLOAT_TYPE r0 = params[0];
	FLOAT_TYPE dx0 = params[1];
	FLOAT_TYPE dy0 = params[2];
	FLOAT_TYPE r1 = params[3];
	FLOAT_TYPE dx1 = params[4];
	FLOAT_TYPE dy1 = params[5];
	FLOAT_TYPE dx2_0 = params[6];
	FLOAT_TYPE dx3_0 = params[7];
	FLOAT_TYPE dx4_0 = params[8];
	FLOAT_TYPE dx2_1 = params[9];
	FLOAT_TYPE dx3_1 = params[10];
	FLOAT_TYPE dx4_1 = params[11];
	FLOAT_TYPE dy2_0 = params[12];
	FLOAT_TYPE dy3_0 = params[13];
	FLOAT_TYPE dy4_0 = params[14];
	FLOAT_TYPE dy2_1 = params[15];
	FLOAT_TYPE dy3_1 = params[16];
	FLOAT_TYPE dy4_1 = params[17];
	FLOAT_TYPE r2_0 = params[18];
	FLOAT_TYPE r3_0 = params[19];
	FLOAT_TYPE r4_0 = params[20];
	FLOAT_TYPE r2_1 = params[21];
	FLOAT_TYPE r3_1 = params[22];
	FLOAT_TYPE r4_1 = params[23];


	float startCost = FullModelCostWelsch(corners1, corners2, w, params);

	for(int i=0;i<length;i++){
		FLOAT_TYPE x1 = corners1[i].x;
		FLOAT_TYPE y1 = corners1[i].y;
		FLOAT_TYPE x2 = corners2[i].x;
		FLOAT_TYPE y2 = corners2[i].y;

		FLOAT_TYPE t0 =  r0 + r1*y1;
		FLOAT_TYPE t1 =  -centerX;
		FLOAT_TYPE t2 =  t1 + x1;
		FLOAT_TYPE t3 =  1.0/centerX;
		FLOAT_TYPE t4 =  t2*t3;
		FLOAT_TYPE t5 =  sin(t0);
		FLOAT_TYPE t6 =  cos(t0);
		FLOAT_TYPE t7 =  t4/2;
		FLOAT_TYPE t8 =  r3_0*y1 + r4_0;
		FLOAT_TYPE t9 =  r3_1*y1 + r4_1;
		FLOAT_TYPE t10 =  dy3_1*y1;
		FLOAT_TYPE t11 =  dy3_0*y1 + dy4_0;
		FLOAT_TYPE t12 =  dy4_1 + t10;
		FLOAT_TYPE t13 =  sin(t8);
		FLOAT_TYPE t14 =  sin(t12);
		FLOAT_TYPE t15 =  sin(t11);
		FLOAT_TYPE t16 =  sin(t9);
		FLOAT_TYPE t17 =  dx3_0*y1;
		FLOAT_TYPE t18 =  dx4_0 + t17;
		FLOAT_TYPE t19 =  dx3_1*y1 + dx4_1;
		FLOAT_TYPE t20 =  r2_0*t13;
		FLOAT_TYPE t21 =  r2_1*t16;
		FLOAT_TYPE t22 =  t5*y1;
		FLOAT_TYPE t23 =  -y2;
		FLOAT_TYPE t24 =  dy2_1*t14;
		FLOAT_TYPE t25 =  t6*y1;
		FLOAT_TYPE t26 =  dy1*y1;
		FLOAT_TYPE t27 =  t20*t7;
		FLOAT_TYPE t28 =  dy0 + dy2_0*t15 + t21*t7 + t23 + t24 + t25 + t26 + t27 + t5*x1;
		FLOAT_TYPE t29 =  pow(w, -2);
		FLOAT_TYPE t30 =  sin(t18);
		FLOAT_TYPE t31 =  sin(t19);
		FLOAT_TYPE t32 =  -t22;
		FLOAT_TYPE t33 =  t6*x1;
		FLOAT_TYPE t34 =  dx2_0*t30;
		FLOAT_TYPE t35 =  -x2;
		FLOAT_TYPE t36 =  dx2_1*t31;
		FLOAT_TYPE t37 =  dx0 + dx1*y1 + t32 + t33 + t34 + t35 + t36;
		FLOAT_TYPE t38 =  (t28*t28);
		FLOAT_TYPE t39 =  (t37*t37);
		FLOAT_TYPE t40 =  exp(t29*(-t38 - t39));
		FLOAT_TYPE t41 =  t29*t40;
		FLOAT_TYPE t42 =  t28*t41;
		FLOAT_TYPE t43 =  (y1*y1);
		FLOAT_TYPE t44 =  -2*y1;
		FLOAT_TYPE t45 =  -2*x1;
		FLOAT_TYPE t46 =  t37*t42;
		FLOAT_TYPE t47 =  2*y1;
		FLOAT_TYPE t48 =  t38*t41;
		FLOAT_TYPE t49 =  cos(t11);
		FLOAT_TYPE t50 =  cos(t8);
		FLOAT_TYPE t51 =  cos(t9);
		FLOAT_TYPE t52 =  cos(t12);
		FLOAT_TYPE t53 =  dy2_1*t52;
		FLOAT_TYPE t54 =  dy2_0*t49;
		FLOAT_TYPE t55 =  t6*x1;
		FLOAT_TYPE t56 =  r2_0*t50;
		FLOAT_TYPE t57 =  r2_1*t51;
		FLOAT_TYPE t58 =  -2*t43;
		FLOAT_TYPE t59 =  cos(t19);
		FLOAT_TYPE t60 =  cos(t18);
		FLOAT_TYPE t61 =  dx2_1*t59;
		FLOAT_TYPE t62 =  dx2_0*t60;
		FLOAT_TYPE t63 =  -t28;
		FLOAT_TYPE t64 =  t45*t5;
		FLOAT_TYPE t65 =  t44*t6;
		FLOAT_TYPE t66 =  -t37;
		FLOAT_TYPE t67 =  -t4;
		FLOAT_TYPE t68 =  2*t40;
		FLOAT_TYPE t69 =  t40*y1;
		FLOAT_TYPE t70 =  t4*t46;
		FLOAT_TYPE t71 =  t40*t47;
		FLOAT_TYPE t72 =  (t2*t2)/pow(centerX, 2);
		FLOAT_TYPE t73 =  t47*t55;
		FLOAT_TYPE t74 =  -2*t22 + 2*t55;
		FLOAT_TYPE t75 =  t5*t58;
		FLOAT_TYPE t76 =  t73 + t75;
		FLOAT_TYPE t77 =  t22*t45 + t58*t6;
		FLOAT_TYPE t78 =  t64 + t65;
		FLOAT_TYPE t79 =  -2*t15;
		FLOAT_TYPE t80 =  -2*t14;
		FLOAT_TYPE t81 =  t37*t41;
		FLOAT_TYPE t82 =  dx2_1*t31;
		FLOAT_TYPE t83 =  dx2_0*t30;
		FLOAT_TYPE t84 =  2*t22;
		FLOAT_TYPE t85 =  t45*t6;
		FLOAT_TYPE t86 =  t2*t40;
		FLOAT_TYPE t87 =  t3*t86;
		FLOAT_TYPE t88 =  -4*t39;
		FLOAT_TYPE t89 =  t41*t88;
		FLOAT_TYPE t90 =  -4*t48;
		FLOAT_TYPE t91 =  -2*t82;
		FLOAT_TYPE t92 =  -2*dx0 + dx1*t44 - 2*t83 + t84 + t85 + t91 + 2*x2;
		FLOAT_TYPE t93 =  t63*t74;
		FLOAT_TYPE t94 =  t21*t67;
		FLOAT_TYPE t95 =  -2*dy0 + dy1*t44 + dy2_0*t79 + dy2_1*t80 + t20*t67 + t64 + t65 + t94 + 2*y2;
		FLOAT_TYPE t96 =  t63*t76 + t66*t77;
		FLOAT_TYPE t97 =  t46*y1;
		FLOAT_TYPE t98 =  t66*t78 + t93;
		FLOAT_TYPE t99 =  t3*t48;
		FLOAT_TYPE t100 =  t4*t57;
		FLOAT_TYPE t101 =  t2*t99;
		FLOAT_TYPE t102 =  t4*t56;
		FLOAT_TYPE t103 =  t44*t70;
		FLOAT_TYPE t104 =  t72/2;
		FLOAT_TYPE t105 =  -t72;
		FLOAT_TYPE t106 =  t105*t48;
		FLOAT_TYPE t107 =  t3*t69;
		FLOAT_TYPE t108 =  t107*t2;
		FLOAT_TYPE t109 =  t101*t44;
		FLOAT_TYPE t110 =  t42*t95;
		FLOAT_TYPE t111 =  t42*t92;
		FLOAT_TYPE t112 =  t42*t96;
		FLOAT_TYPE t113 =  t42*t98;
		FLOAT_TYPE t114 =  -t5;
		FLOAT_TYPE t115 =  t47*t81;
		FLOAT_TYPE t116 =  t89*y1;
		FLOAT_TYPE t117 =  2*t81;
		FLOAT_TYPE t118 =  t90*y1;
		FLOAT_TYPE t119 =  t16*t4;
		FLOAT_TYPE t120 =  t13*t4;
		FLOAT_TYPE t121 =  t104*t40;
		FLOAT_TYPE t122 =  t43*t53;
		FLOAT_TYPE t123 =  -t6;
		FLOAT_TYPE t124 =  t104*t69;
		FLOAT_TYPE t125 =  t61*t62;
		FLOAT_TYPE t126 =  t106*y1;
		FLOAT_TYPE t127 =  t56*t57;
		FLOAT_TYPE t128 =  t114*t43 + t55*y1;
		FLOAT_TYPE t129 =  t32 + t33;
		FLOAT_TYPE t130 =  t46*t62;
		FLOAT_TYPE t131 =  t54*t69;
		FLOAT_TYPE t132 =  t46*t61;
		FLOAT_TYPE t133 =  t56*t70;
		FLOAT_TYPE t134 =  t57*t70;
		FLOAT_TYPE t135 =  t43*t54;
		FLOAT_TYPE t136 =  -t41;
		FLOAT_TYPE t137 =  t53*t54;
		FLOAT_TYPE t138 =  t53*t97;
		FLOAT_TYPE t139 =  t103*t62;
		FLOAT_TYPE t140 =  t109*t54;
		FLOAT_TYPE t141 =  t102*t48;
		FLOAT_TYPE t142 =  -t40;
		FLOAT_TYPE t143 =  t28*t40;
		FLOAT_TYPE t144 =  t54*t97;
		FLOAT_TYPE t145 =  t103*t61;
		FLOAT_TYPE t146 =  t57*t87;
		FLOAT_TYPE t147 =  t22*x1;
		FLOAT_TYPE t148 =  -2*t48;
		FLOAT_TYPE t149 =  t101*t58;
		FLOAT_TYPE t150 =  t13*t87;
		FLOAT_TYPE t151 =  t16*t87;
		FLOAT_TYPE t152 =  t43*t68;
		FLOAT_TYPE t153 =  t123*t43;
		FLOAT_TYPE t154 =  2*t112;
		FLOAT_TYPE t155 =  t15*t68;
		FLOAT_TYPE t156 =  -t147;
		FLOAT_TYPE t157 =  t153 + t156;
		FLOAT_TYPE t158 =  t114*x1 + t123*y1;
		FLOAT_TYPE t159 =  t31*t68;
		FLOAT_TYPE t160 =  t102*t40;
		FLOAT_TYPE t161 =  t13*t57;
		FLOAT_TYPE t162 =  2*t110;
		FLOAT_TYPE t163 =  t43*t89;
		FLOAT_TYPE t164 =  t16*t70;
		FLOAT_TYPE t165 =  (dx2_1*dx2_1)*(t59*t59);
		FLOAT_TYPE t166 =  t14*t54;
		FLOAT_TYPE t167 =  (t51*t51);
		FLOAT_TYPE t168 =  t117*t92;
		FLOAT_TYPE t169 =  (r2_0*r2_0)*(t50*t50);
		FLOAT_TYPE t170 =  t115*t62;
		FLOAT_TYPE t171 =  t30*t62;
		FLOAT_TYPE t172 =  (dy2_0*dy2_0);
		FLOAT_TYPE t173 =  t31*t46;
		FLOAT_TYPE t174 =  t47*t54;
		FLOAT_TYPE t175 =  t53*t69;
		FLOAT_TYPE t176 =  (dy2_1*dy2_1);
		FLOAT_TYPE t177 =  t117*t95;
		FLOAT_TYPE t178 =  2*t113;
		FLOAT_TYPE t179 =  t176*(t52*t52);
		FLOAT_TYPE t180 =  t117*t96;
		FLOAT_TYPE t181 =  t56*t87;
		FLOAT_TYPE t182 =  t16*t56;
		FLOAT_TYPE t183 =  t14*t53;
		FLOAT_TYPE t184 =  (r2_1*r2_1);
		FLOAT_TYPE t185 =  t167*t184;
		FLOAT_TYPE t186 =  t53*t56;
		FLOAT_TYPE t187 =  (dx2_0*dx2_0)*(t60*t60);
		FLOAT_TYPE t188 =  t172*(t49*t49);
		FLOAT_TYPE t189 =  t13*t70;
		FLOAT_TYPE t190 =  t115*t61;
		FLOAT_TYPE t191 =  t30*t46;
		FLOAT_TYPE t192 =  t37*t40;
		FLOAT_TYPE t193 =  t117*t98;
		FLOAT_TYPE t194 =  t100*y1;
		FLOAT_TYPE t195 =  t30*t61;
		FLOAT_TYPE t196 =  t3*t56;
		FLOAT_TYPE t197 =  t28*t68;
		FLOAT_TYPE t198 =  t196*t2;
		FLOAT_TYPE t199 =  t37*t68;
		FLOAT_TYPE t200 =  t47*t53;
		FLOAT_TYPE t201 =  t102*t69;
		FLOAT_TYPE t202 =  t100*t40;
		FLOAT_TYPE t203 =  2*t5;
		FLOAT_TYPE t204 =  t31*t89;
		FLOAT_TYPE t205 =  t13*t56;
		FLOAT_TYPE t206 =  t109*t57;
		FLOAT_TYPE t207 =  t16*t57;
		FLOAT_TYPE t208 =  t100*t48;
		FLOAT_TYPE t209 =  t136*t95;
		FLOAT_TYPE t210 =  t15*t90;
		FLOAT_TYPE t211 =  t28*t71;
		FLOAT_TYPE t212 =  t148*t53;
		FLOAT_TYPE t213 =  t44*t53;
		FLOAT_TYPE t214 =  t120*t48;
		FLOAT_TYPE t215 =  t108*t57;
		FLOAT_TYPE t216 =  t37*t71;
		FLOAT_TYPE t217 =  t40*t74;
		FLOAT_TYPE t218 =  t40*t76;
		FLOAT_TYPE t219 =  t102*y1;
		FLOAT_TYPE t220 =  t106*t43;
		FLOAT_TYPE t221 =  t58*t61;
		FLOAT_TYPE t222 =  t28*t51;
		FLOAT_TYPE t223 =  t108*t186 + t141*t213;
		FLOAT_TYPE t224 =  t192*t44;
		FLOAT_TYPE t225 =  t116*t125;
		FLOAT_TYPE t226 =  t125*t71 + t225;
		FLOAT_TYPE t227 =  t13*t16;
		FLOAT_TYPE t228 =  t118*t137 + t137*t71;
		FLOAT_TYPE t229 =  t102*t131;
		FLOAT_TYPE t230 =  t140*t56;
		FLOAT_TYPE t231 =  t229 + t230;
		FLOAT_TYPE t232 =  t31*t62;
		FLOAT_TYPE t233 =  -2*t54;
		FLOAT_TYPE t234 =  -4*t144*t61;
		FLOAT_TYPE t235 =  t74/2;
		FLOAT_TYPE t236 =  t149*t57;
		FLOAT_TYPE t237 =  t76/2;
		FLOAT_TYPE t238 =  t124*t127;
		FLOAT_TYPE t239 =  t61*t71;
		FLOAT_TYPE t240 =  dy2_1*t143;
		FLOAT_TYPE t241 =  t15*t44;
		FLOAT_TYPE t242 =  t109*t56;
		FLOAT_TYPE t243 =  t44*t55;
		FLOAT_TYPE t244 =  t103*t57;
		FLOAT_TYPE t245 =  t136*t98;
		FLOAT_TYPE t246 =  t28*t50;
		FLOAT_TYPE t247 =  t101*t79;
		FLOAT_TYPE t248 =  -4*t138*t61;
		FLOAT_TYPE t249 =  t54*t71;
		FLOAT_TYPE t250 =  t15*t53;
		FLOAT_TYPE t251 =  t148*t54;
		FLOAT_TYPE t252 =  t203*t43;
		FLOAT_TYPE t253 =  t243 + t252;
		FLOAT_TYPE t254 =  t108*t56;
		FLOAT_TYPE t255 =  t119*t48;
		FLOAT_TYPE t256 =  t206*t53;
		FLOAT_TYPE t257 =  -4*t138*t62;
		FLOAT_TYPE t258 =  t100*t131;
		FLOAT_TYPE t259 =  t126*t127 + t238;
		FLOAT_TYPE t260 =  t14*t97;
		FLOAT_TYPE t261 =  t53*t71;
		FLOAT_TYPE t262 =  t62*t71;
		FLOAT_TYPE t263 =  t108*t13;
		FLOAT_TYPE t264 =  -4*t144*t62;
		FLOAT_TYPE t265 =  t103*t56;
		FLOAT_TYPE t266 =  t140*t57 + t258;
		FLOAT_TYPE t267 =  t139*t56;
		FLOAT_TYPE t268 =  t145*t56;
		FLOAT_TYPE t269 =  t58*t62;
		FLOAT_TYPE t270 =  t100*t175;
		FLOAT_TYPE t271 =  t256 + t270;
		FLOAT_TYPE t272 =  t40*t78;
		FLOAT_TYPE t273 =  t40*t77;
		FLOAT_TYPE t274 =  t15*t97;
		FLOAT_TYPE t275 =  t121*t43;
		FLOAT_TYPE t276 =  dy2_0*t143;
		FLOAT_TYPE t277 =  t139*t57;
		FLOAT_TYPE t278 =  t54*t68;
		FLOAT_TYPE t279 =  t145*t57;

		jacob(0, 0) +=  t136*(t98*t98) - t142*(t129*t74 + t158*t78 - t63*t78 - t66*(t84 + t85));
		jacob(0, 3) +=  -t142*(t128*t74 + t157*t78 - t253*t66 - t63*t77) + t245*t96;
		jacob(0, 4) +=  t115*t98 + t69*t78;
		jacob(0, 5) +=  t113*t47 + t69*t74;
		jacob(0, 6) +=  t30*(t193 + t272);
		jacob(0, 7) +=  t170*t98 + t62*t69*t78;
		jacob(0, 8) +=  t62*(t193 + t272);
		jacob(0, 9) +=  t31*(t193 + t272);
		jacob(0, 10) +=  t190*t98 + t61*t69*t78;
		jacob(0, 11) +=  t61*(t193 + t272);
		jacob(0, 12) +=  t15*(t178 + t217);
		jacob(0, 13) +=  t113*t174 + t131*t74;
		jacob(0, 14) +=  t54*(t178 + t217);
		jacob(0, 15) +=  t14*(t178 + t217);
		jacob(0, 16) +=  t113*t200 + t175*t74;
		jacob(0, 17) +=  t53*(t178 + t217);
		jacob(0, 18) +=  t13*(t113*t2*t3 + t217*t7);
		jacob(0, 19) +=  t108*t235*t56 + t113*t219;
		jacob(0, 20) +=  t113*t198 + t217*t56*t7;
		jacob(0, 21) +=  t113*t119 + t151*t235;
		jacob(0, 22) +=  t113*t194 + t215*t235;
		jacob(0, 23) +=  t100*t113 + t146*t235;
		jacob(1, 0) +=  t142*(t203*x1 + t47*t6) + t245*t92;
		jacob(1, 1) +=  t136*(t92*t92) + t68;
		jacob(2, 0) +=  t142*(t84 + t85) + t209*t98;
		jacob(2, 1) +=  t209*t92;
		jacob(2, 2) +=  t136*(t95*t95) + t68;
		jacob(3, 0) +=  -t142*(t129*t76 + t158*t77 - t253*t66 - t63*t77) + t245*t96;
		jacob(3, 1) +=  t136*t92*t96 + 2*t142*(t147 + t43*t6);
		jacob(3, 2) +=  t142*t253 + t209*t96;
		jacob(3, 3) +=  t136*(t96*t96) - t142*(t128*t76 + t157*t77 - t63*(t43*t45*t5 - 2*t6*pow(y1, 3)) - t66*(t203*pow(y1, 3) + t43*t45*t6));
		jacob(3, 4) +=  t115*t96 + t69*t77;
		jacob(3, 5) +=  t112*t47 + t69*t76;
		jacob(3, 6) +=  t30*(t180 + t273);
		jacob(3, 7) +=  t170*t96 + t62*t69*t77;
		jacob(3, 8) +=  t62*(t180 + t273);
		jacob(3, 9) +=  t31*(t180 + t273);
		jacob(3, 10) +=  t190*t96 + t61*t69*t77;
		jacob(3, 11) +=  t61*(t180 + t273);
		jacob(3, 12) +=  t15*(t154 + t218);
		jacob(3, 13) +=  t112*t174 + t131*t76;
		jacob(3, 14) +=  t54*(t154 + t218);
		jacob(3, 15) +=  t14*(t154 + t218);
		jacob(3, 16) +=  t154*t53*y1 + t175*t76;
		jacob(3, 17) +=  t53*(t154 + t218);
		jacob(3, 18) +=  t112*t120 + t150*t237;
		jacob(3, 19) +=  t112*t198*y1 + t56*t69*t7*t76;
		jacob(3, 20) +=  t102*t112 + t181*t237;
		jacob(3, 21) +=  t16*(t112*t2*t3 + t218*t7);
		jacob(3, 22) +=  t112*t194 + t215*t237;
		jacob(3, 23) +=  t100*t112 + t146*t237;
		jacob(4, 0) +=  t115*t98 + t158*t71;
		jacob(4, 1) +=  t115*t92 + t71;
		jacob(4, 2) +=  t115*t95;
		jacob(4, 3) +=  t115*t96 + t157*t71;
		jacob(4, 4) +=  t152 + t163;
		jacob(5, 0) +=  t113*t47 + t129*t71;
		jacob(5, 1) +=  t111*t47;
		jacob(5, 2) +=  t110*t47 + t71;
		jacob(5, 3) +=  t112*t47 + t128*t71;
		jacob(5, 4) +=  t43*t46;
		jacob(5, 5) +=  t152 + t43*t90;
		jacob(6, 0) +=  t30*(t158*t68 + t193);
		jacob(6, 1) +=  t30*(t168 + t68);
		jacob(6, 2) +=  t177*t30;
		jacob(6, 3) +=  t30*(t157*t68 + t180);
		jacob(6, 4) +=  t30*(t116 + t71);
		jacob(6, 5) +=  t30*t97;
		jacob(6, 6) +=  (t30*t30)*(t68 + t89);
		jacob(7, 0) +=  t158*t62*t71 + t170*t98;
		jacob(7, 1) +=  t170*t92 + t262;
		jacob(7, 2) +=  t170*t95;
		jacob(7, 3) +=  t157*t62*t71 + t170*t96;
		jacob(7, 4) +=  t62*(t152 + t163);
		jacob(7, 5) +=  t130*t43;
		jacob(7, 6) +=  t116*t171 + t171*t71 + t216*t60;
		jacob(7, 7) +=  t152*t187 + t163*t187 + t192*t58*t83;
		jacob(8, 0) +=  t62*(t158*t68 + t193);
		jacob(8, 1) +=  t62*(t168 + t68);
		jacob(8, 2) +=  t177*t62;
		jacob(8, 3) +=  t62*(t157*t68 + t180);
		jacob(8, 4) +=  t116*t62 + t262;
		jacob(8, 5) +=  t62*t97;
		jacob(8, 6) +=  t171*t68 + t171*t89 + t199*t60;
		jacob(8, 7) +=  t116*t187 + t187*t71 + t224*t83;
		jacob(8, 8) +=  t187*t68 + t187*t89 - 2*t192*t83;
		jacob(9, 0) +=  t158*t159 + t193*t31;
		jacob(9, 1) +=  t159 + t168*t31;
		jacob(9, 2) +=  t177*t31;
		jacob(9, 3) +=  t157*t159 + t180*t31;
		jacob(9, 4) +=  t31*(t116 + t71);
		jacob(9, 5) +=  t31*t97;
		jacob(9, 6) +=  t30*(t159 + t204);
		jacob(9, 7) +=  t232*(t116 + t71);
		jacob(9, 8) +=  t62*(t159 + t204);
		jacob(9, 9) +=  (t31*t31)*(t68 + t89);
		jacob(10, 0) +=  t158*t239 + t190*t98;
		jacob(10, 1) +=  t190*t92 + t239;
		jacob(10, 2) +=  t190*t95;
		jacob(10, 3) +=  t157*t239 + t190*t96;
		jacob(10, 4) +=  t61*(t152 + t163);
		jacob(10, 5) +=  t132*t43;
		jacob(10, 6) +=  t195*(t116 + t71);
		jacob(10, 7) +=  t125*(t152 + t163);
		jacob(10, 9) +=  t116*t31*t61 + t216*t59 + t239*t31;
		jacob(10, 10) +=  t152*t165 + t163*t165 + t192*t58*t82;
		jacob(11, 0) +=  t61*(t158*t68 + t193);
		jacob(11, 1) +=  t61*(t168 + t68);
		jacob(11, 2) +=  t177*t61;
		jacob(11, 3) +=  t61*(t157*t68 + t180);
		jacob(11, 4) +=  t116*t61 + t239;
		jacob(11, 5) +=  t61*t97;
		jacob(11, 6) +=  t195*(t68 + t89);
		jacob(11, 7) +=  t226;
		jacob(11, 8) +=  t125*(t68 + t89);
		jacob(11, 9) +=  t159*t61 + t199*t59 + t204*t61;
		jacob(11, 10) +=  t116*t165 + t165*t71 + t224*t82;
		jacob(11, 11) +=  t165*t68 + t165*t89 - 2*t192*t82;
		jacob(12, 0) +=  t129*t155 + t15*t178;
		jacob(12, 1) +=  t111*t15;
		jacob(12, 2) +=  t15*t162 + t155;
		jacob(12, 3) +=  t128*t155 + t15*t154;
		jacob(12, 4) +=  t274;
		jacob(12, 5) +=  t15*(t118 + t71);
		jacob(12, 6) +=  t15*t191;
		jacob(12, 7) +=  t274*t62;
		jacob(12, 8) +=  t130*t15;
		jacob(12, 9) +=  t15*t173;
		jacob(12, 10) +=  t274*t61;
		jacob(12, 11) +=  t132*t15;
		jacob(12, 12) +=  (t15*t15)*(t68 + t90);
		jacob(13, 0) +=  t113*t174 + t129*t249;
		jacob(13, 1) +=  t111*t174;
		jacob(13, 2) +=  t110*t174 + t249;
		jacob(13, 3) +=  t112*t174 + t128*t249;
		jacob(13, 4) +=  t135*t46;
		jacob(13, 5) +=  t135*(t68 + t90);
		jacob(13, 6) +=  t144*t30;
		jacob(13, 7) +=  t130*t135;
		jacob(13, 9) +=  t144*t31;
		jacob(13, 10) +=  t132*t135;
		jacob(13, 12) +=  t118*t15*t54 + t15*t249 + t211*t49;
		jacob(13, 13) +=  t15*t276*t58 + t152*t188 + t188*t43*t90;
		jacob(14, 0) +=  t129*t278 + t178*t54;
		jacob(14, 1) +=  t111*t54;
		jacob(14, 2) +=  t162*t54 + t278;
		jacob(14, 3) +=  t128*t278 + t154*t54;
		jacob(14, 4) +=  t144;
		jacob(14, 5) +=  t118*t54 + t249;
		jacob(14, 6) +=  t191*t54;
		jacob(14, 7) +=  t264;
		jacob(14, 8) +=  t130*t54;
		jacob(14, 9) +=  t173*t54;
		jacob(14, 10) +=  t234;
		jacob(14, 11) +=  t132*t54;
		jacob(14, 12) +=  t155*t54 + t197*t49 + t210*t54;
		jacob(14, 13) +=  t118*t188 + t188*t71 + t241*t276;
		jacob(14, 14) +=  t188*t68 + t188*t90 + t276*t79;
		jacob(15, 0) +=  t14*(t129*t68 + t178);
		jacob(15, 1) +=  t111*t14;
		jacob(15, 2) +=  t14*(t162 + t68);
		jacob(15, 3) +=  t14*(t128*t68 + t154);
		jacob(15, 4) +=  t260;
		jacob(15, 5) +=  t14*(t118 + t71);
		jacob(15, 6) +=  t14*t191;
		jacob(15, 7) +=  t260*t62;
		jacob(15, 8) +=  t130*t14;
		jacob(15, 9) +=  t14*t173;
		jacob(15, 10) +=  t260*t61;
		jacob(15, 11) +=  t132*t14;
		jacob(15, 12) +=  t14*(t155 + t210);
		jacob(15, 13) +=  t166*(t118 + t71);
		jacob(15, 14) +=  t166*(t68 + t90);
		jacob(15, 15) +=  (t14*t14)*(t68 + t90);
		jacob(16, 0) +=  t113*t200 + t129*t53*t71;
		jacob(16, 1) +=  t111*t200;
		jacob(16, 2) +=  t110*t200 + t261;
		jacob(16, 3) +=  t112*t200 + t128*t53*t71;
		jacob(16, 4) +=  t122*t46;
		jacob(16, 5) +=  t122*(t68 + t90);
		jacob(16, 6) +=  t138*t30;
		jacob(16, 7) +=  t122*t130;
		jacob(16, 9) +=  t138*t31;
		jacob(16, 10) +=  t122*t132;
		jacob(16, 12) +=  t250*(t118 + t71);
		jacob(16, 13) +=  t122*(t278 + t54*t90);
		jacob(16, 15) +=  t118*t183 + t183*t71 + t211*t52;
		jacob(16, 16) +=  t14*t240*t58 + t152*t179 + t179*t43*t90;
		jacob(17, 0) +=  t53*(t129*t68 + t178);
		jacob(17, 1) +=  t111*t53;
		jacob(17, 2) +=  t53*(t162 + t68);
		jacob(17, 3) +=  t53*(t128*t68 + t154);
		jacob(17, 4) +=  t138;
		jacob(17, 5) +=  t118*t53 + t261;
		jacob(17, 6) +=  t191*t53;
		jacob(17, 7) +=  t257;
		jacob(17, 8) +=  t130*t53;
		jacob(17, 9) +=  t173*t53;
		jacob(17, 10) +=  t248;
		jacob(17, 11) +=  t132*t53;
		jacob(17, 12) +=  t53*(t155 + t210);
		jacob(17, 13) +=  t228;
		jacob(17, 14) +=  t137*(t68 + t90);
		jacob(17, 15) +=  t183*t68 + t183*t90 + t197*t52;
		jacob(17, 16) +=  t118*t179 + t14*t240*t44 + t179*t71;
		jacob(17, 17) +=  t179*t68 + t179*t90 + t240*t80;
		jacob(18, 0) +=  t113*t120 + t129*t150;
		jacob(18, 1) +=  t111*t120;
		jacob(18, 2) +=  t110*t120 + t150;
		jacob(18, 3) +=  t112*t120 + t128*t150;
		jacob(18, 4) +=  t103*t13;
		jacob(18, 5) +=  t109*t13 + t120*t69;
		jacob(18, 6) +=  t189*t30;
		jacob(18, 7) +=  t13*t139;
		jacob(18, 8) +=  t189*t62;
		jacob(18, 9) +=  t189*t31;
		jacob(18, 10) +=  t13*t145;
		jacob(18, 11) +=  t189*t61;
		jacob(18, 12) +=  t15*t150 + t214*t79;
		jacob(18, 13) +=  t120*t131 + t13*t140;
		jacob(18, 14) +=  t101*t13*t233 + t120*t40*t54;
		jacob(18, 15) +=  t14*t150 + t214*t80;
		jacob(18, 16) +=  t213*t214 + t263*t53;
		jacob(18, 17) +=  t120*t212 + t150*t53;
		jacob(18, 18) +=  (t13*t13)*(t106 + t121);
		jacob(19, 0) +=  t113*t198*y1 + t129*t201;
		jacob(19, 1) +=  t111*t219;
		jacob(19, 2) +=  t110*t219 + t254;
		jacob(19, 3) +=  t112*t198*y1 + t128*t201;
		jacob(19, 4) +=  t133*t58;
		jacob(19, 5) +=  t149*t56 + t160*t43;
		jacob(19, 6) +=  t265*t30;
		jacob(19, 7) +=  t133*t269;
		jacob(19, 9) +=  t265*t31;
		jacob(19, 10) +=  t133*t221;
		jacob(19, 12) +=  t15*(t201 + t242);
		jacob(19, 13) +=  t135*t181 + t141*t54*t58;
		jacob(19, 15) +=  t14*(t201 + t242);
		jacob(19, 16) +=  t122*t160 + t149*t186;
		jacob(19, 18) +=  t108*t246 + t124*t205 + t126*t205;
		jacob(19, 19) +=  t169*t220 + t169*t275 + t20*t43*t63*t87;
		jacob(20, 0) +=  t113*t198 + t129*t160;
		jacob(20, 1) +=  t102*t111;
		jacob(20, 2) +=  t110*t198 + t160;
		jacob(20, 3) +=  t102*t112 + t128*t181;
		jacob(20, 4) +=  t265;
		jacob(20, 5) +=  t141*t44 + t254;
		jacob(20, 6) +=  t133*t30;
		jacob(20, 7) +=  t267;
		jacob(20, 8) +=  t133*t62;
		jacob(20, 9) +=  t133*t31;
		jacob(20, 10) +=  t268;
		jacob(20, 11) +=  t133*t61;
		jacob(20, 12) +=  t15*t160 + t247*t56;
		jacob(20, 13) +=  t231;
		jacob(20, 14) +=  t141*t233 + t181*t54;
		jacob(20, 15) +=  t14*t181 + t141*t80;
		jacob(20, 16) +=  t223;
		jacob(20, 17) +=  -2*t101*t186 + t160*t53;
		jacob(20, 18) +=  t106*t20*t50 + t121*t205 + t246*t87;
		jacob(20, 19) +=  r2_0*t263*t63 + t124*t169 + t126*t169;
		jacob(20, 20) +=  r2_0*t150*t63 + t106*t169 + t121*t169;
		jacob(21, 0) +=  t113*t119 + t129*t151;
		jacob(21, 1) +=  t111*t119;
		jacob(21, 2) +=  t110*t119 + t151;
		jacob(21, 3) +=  t112*t16*t2*t3 + t119*t128*t40;
		jacob(21, 4) +=  t103*t16;
		jacob(21, 5) +=  t108*t16 + t255*t44;
		jacob(21, 6) +=  t164*t30;
		jacob(21, 7) +=  t139*t16;
		jacob(21, 8) +=  t164*t62;
		jacob(21, 9) +=  t164*t31;
		jacob(21, 10) +=  t145*t16;
		jacob(21, 11) +=  t164*t61;
		jacob(21, 12) +=  t119*t15*t40 + t16*t247;
		jacob(21, 13) +=  t119*t131 + t140*t16;
		jacob(21, 14) +=  t119*t251 + t151*t54;
		jacob(21, 15) +=  t14*t151 + t255*t80;
		jacob(21, 16) +=  t109*t16*t53 + t119*t175;
		jacob(21, 17) +=  t119*t212 + t151*t53;
		jacob(21, 18) +=  t227*(t106 + t121);
		jacob(21, 19) +=  t182*(t124 + t126);
		jacob(21, 20) +=  t182*(t106 + t121);
		jacob(21, 21) +=  (t16*t16)*(t106 + t121);
		jacob(22, 0) +=  t113*t194 + t129*t215;
		jacob(22, 1) +=  t111*t194;
		jacob(22, 2) +=  t110*t194 + t215;
		jacob(22, 3) +=  t112*t194 + t128*t215;
		jacob(22, 4) +=  t134*t58;
		jacob(22, 5) +=  t202*t43 + t236;
		jacob(22, 6) +=  t244*t30;
		jacob(22, 7) +=  t134*t269;
		jacob(22, 9) +=  t244*t31;
		jacob(22, 10) +=  t134*t221;
		jacob(22, 12) +=  t15*t215 + t208*t241;
		jacob(22, 13) +=  t135*t202 + t236*t54;
		jacob(22, 15) +=  t14*(t100*t69 + t206);
		jacob(22, 16) +=  t122*t202 + t236*t53;
		jacob(22, 18) +=  t161*(t124 + t126);
		jacob(22, 19) +=  t127*(t220 + t275);
		jacob(22, 21) +=  t108*t222 + t124*t207 + t126*t207;
		jacob(22, 22) +=  r2_1*t151*t43*t63 + t185*t220 + t185*t275;
		jacob(23, 0) +=  t113*t2*t3*t57 + t129*t202;
		jacob(23, 1) +=  t100*t111;
		jacob(23, 2) +=  t100*t110 + t146;
		jacob(23, 3) +=  t100*t112 + t128*t146;
		jacob(23, 4) +=  t244;
		jacob(23, 5) +=  t100*t69 + t206;
		jacob(23, 6) +=  t134*t30;
		jacob(23, 7) +=  t277;
		jacob(23, 8) +=  t134*t62;
		jacob(23, 9) +=  t134*t31;
		jacob(23, 10) +=  t279;
		jacob(23, 11) +=  t134*t61;
		jacob(23, 12) +=  t146*t15 + t208*t79;
		jacob(23, 13) +=  t266;
		jacob(23, 14) +=  t100*t251 + t146*t54;
		jacob(23, 15) +=  t14*t146 + t208*t80;
		jacob(23, 16) +=  t271;
		jacob(23, 17) +=  t100*t212 + t146*t53;
		jacob(23, 18) +=  t161*(t106 + t121);
		jacob(23, 19) +=  t259;
		jacob(23, 20) +=  t127*(t106 + t121);
		jacob(23, 21) +=  t106*t21*t51 + t121*t207 + t222*t87;
		jacob(23, 22) +=  t108*t21*t63 + t124*t185 + t126*t185;
		jacob(23, 23) +=  r2_1*t151*t63 + t106*t185 + t121*t185;

		fVector(0) +=  t142*t98;
		fVector(1) +=  t142*t92;
		fVector(2) +=  t142*t95;
		fVector(3) +=  t142*t96;
		fVector(4) +=  t216;
		fVector(5) +=  t211;
		fVector(6) +=  t199*t30;
		fVector(7) +=  t216*t62;
		fVector(8) +=  t199*t62;
		fVector(9) +=  t159*t37;
		fVector(10) +=  t216*t61;
		fVector(11) +=  t199*t61;
		fVector(12) +=  t155*t28;
		fVector(13) +=  t211*t54;
		fVector(14) +=  t197*t54;
		fVector(15) +=  t14*t197;
		fVector(16) +=  t211*t53;
		fVector(17) +=  t197*t53;
		fVector(18) +=  t120*t143;
		fVector(19) +=  t201*t28;
		fVector(20) +=  t102*t143;
		fVector(21) +=  t119*t143;
		fVector(22) +=  t100*t28*t69;
		fVector(23) +=  t100*t143;

	}

	jacob(5, 4) *= -4.000000;
	jacob(6, 5) *= -4.000000;
	jacob(7, 5) *= -4.000000;
	jacob(8, 5) *= -4.000000;
	jacob(9, 5) *= -4.000000;
	jacob(10, 5) *= -4.000000;
	jacob(11, 5) *= -4.000000;
	jacob(12, 1) *= 2.000000;
	jacob(12, 4) *= -4.000000;
	jacob(12, 6) *= -4.000000;
	jacob(12, 7) *= -4.000000;
	jacob(12, 8) *= -4.000000;
	jacob(12, 9) *= -4.000000;
	jacob(12, 10) *= -4.000000;
	jacob(12, 11) *= -4.000000;
	jacob(13, 4) *= -4.000000;
	jacob(13, 6) *= -4.000000;
	jacob(13, 7) *= -4.000000;
	jacob(13, 9) *= -4.000000;
	jacob(13, 10) *= -4.000000;
	jacob(14, 1) *= 2.000000;
	jacob(14, 4) *= -4.000000;
	jacob(14, 6) *= -4.000000;
	jacob(14, 8) *= -4.000000;
	jacob(14, 9) *= -4.000000;
	jacob(14, 11) *= -4.000000;
	jacob(15, 1) *= 2.000000;
	jacob(15, 4) *= -4.000000;
	jacob(15, 6) *= -4.000000;
	jacob(15, 7) *= -4.000000;
	jacob(15, 8) *= -4.000000;
	jacob(15, 9) *= -4.000000;
	jacob(15, 10) *= -4.000000;
	jacob(15, 11) *= -4.000000;
	jacob(16, 4) *= -4.000000;
	jacob(16, 6) *= -4.000000;
	jacob(16, 7) *= -4.000000;
	jacob(16, 9) *= -4.000000;
	jacob(16, 10) *= -4.000000;
	jacob(17, 1) *= 2.000000;
	jacob(17, 4) *= -4.000000;
	jacob(17, 6) *= -4.000000;
	jacob(17, 8) *= -4.000000;
	jacob(17, 9) *= -4.000000;
	jacob(17, 11) *= -4.000000;
	jacob(18, 6) *= -2.000000;
	jacob(18, 8) *= -2.000000;
	jacob(18, 9) *= -2.000000;
	jacob(18, 11) *= -2.000000;
	jacob(20, 6) *= -2.000000;
	jacob(20, 8) *= -2.000000;
	jacob(20, 9) *= -2.000000;
	jacob(20, 11) *= -2.000000;
	jacob(21, 6) *= -2.000000;
	jacob(21, 8) *= -2.000000;
	jacob(21, 9) *= -2.000000;
	jacob(21, 11) *= -2.000000;
	jacob(23, 6) *= -2.000000;
	jacob(23, 8) *= -2.000000;
	jacob(23, 9) *= -2.000000;
	jacob(23, 11) *= -2.000000;


	jacob(0, 1) = jacob(1, 0);
	jacob(0, 2) = jacob(2, 0);
	jacob(1, 2) = jacob(2, 1);
	jacob(1, 3) = jacob(3, 1);
	jacob(1, 4) = jacob(4, 1);
	jacob(1, 5) = jacob(5, 1);
	jacob(1, 6) = jacob(6, 1);
	jacob(1, 7) = jacob(7, 1);
	jacob(1, 8) = jacob(8, 1);
	jacob(1, 9) = jacob(9, 1);
	jacob(1, 10) = jacob(10, 1);
	jacob(1, 11) = jacob(11, 1);
	jacob(1, 12) = jacob(12, 1);
	jacob(1, 13) = jacob(13, 1);
	jacob(1, 14) = jacob(14, 1);
	jacob(1, 15) = jacob(15, 1);
	jacob(1, 16) = jacob(16, 1);
	jacob(1, 17) = jacob(17, 1);
	jacob(1, 18) = jacob(18, 1);
	jacob(1, 19) = jacob(19, 1);
	jacob(1, 20) = jacob(20, 1);
	jacob(1, 21) = jacob(21, 1);
	jacob(1, 22) = jacob(22, 1);
	jacob(1, 23) = jacob(23, 1);
	jacob(2, 3) = jacob(3, 2);
	jacob(2, 4) = jacob(4, 2);
	jacob(2, 5) = jacob(5, 2);
	jacob(2, 6) = jacob(6, 2);
	jacob(2, 7) = jacob(7, 2);
	jacob(2, 8) = jacob(8, 2);
	jacob(2, 9) = jacob(9, 2);
	jacob(2, 10) = jacob(10, 2);
	jacob(2, 11) = jacob(11, 2);
	jacob(2, 12) = jacob(12, 2);
	jacob(2, 13) = jacob(13, 2);
	jacob(2, 14) = jacob(14, 2);
	jacob(2, 15) = jacob(15, 2);
	jacob(2, 16) = jacob(16, 2);
	jacob(2, 17) = jacob(17, 2);
	jacob(2, 18) = jacob(18, 2);
	jacob(2, 19) = jacob(19, 2);
	jacob(2, 20) = jacob(20, 2);
	jacob(2, 21) = jacob(21, 2);
	jacob(2, 22) = jacob(22, 2);
	jacob(2, 23) = jacob(23, 2);
	jacob(4, 5) = jacob(5, 4);
	jacob(4, 6) = jacob(6, 4);
	jacob(4, 7) = jacob(7, 4);
	jacob(4, 8) = jacob(8, 4);
	jacob(4, 9) = jacob(9, 4);
	jacob(4, 10) = jacob(10, 4);
	jacob(4, 11) = jacob(11, 4);
	jacob(4, 12) = jacob(12, 4);
	jacob(4, 13) = jacob(13, 4);
	jacob(4, 14) = jacob(14, 4);
	jacob(4, 15) = jacob(15, 4);
	jacob(4, 16) = jacob(16, 4);
	jacob(4, 17) = jacob(17, 4);
	jacob(4, 18) = jacob(18, 4);
	jacob(4, 19) = jacob(19, 4);
	jacob(4, 20) = jacob(20, 4);
	jacob(4, 21) = jacob(21, 4);
	jacob(4, 22) = jacob(22, 4);
	jacob(4, 23) = jacob(23, 4);
	jacob(5, 6) = jacob(6, 5);
	jacob(5, 7) = jacob(7, 5);
	jacob(5, 8) = jacob(8, 5);
	jacob(5, 9) = jacob(9, 5);
	jacob(5, 10) = jacob(10, 5);
	jacob(5, 11) = jacob(11, 5);
	jacob(5, 12) = jacob(12, 5);
	jacob(5, 13) = jacob(13, 5);
	jacob(5, 14) = jacob(14, 5);
	jacob(5, 15) = jacob(15, 5);
	jacob(5, 16) = jacob(16, 5);
	jacob(5, 17) = jacob(17, 5);
	jacob(5, 18) = jacob(18, 5);
	jacob(5, 19) = jacob(19, 5);
	jacob(5, 20) = jacob(20, 5);
	jacob(5, 21) = jacob(21, 5);
	jacob(5, 22) = jacob(22, 5);
	jacob(5, 23) = jacob(23, 5);
	jacob(6, 7) = jacob(7, 6);
	jacob(6, 8) = jacob(8, 6);
	jacob(6, 9) = jacob(9, 6);
	jacob(6, 10) = jacob(10, 6);
	jacob(6, 11) = jacob(11, 6);
	jacob(6, 12) = jacob(12, 6);
	jacob(6, 13) = jacob(13, 6);
	jacob(6, 14) = jacob(14, 6);
	jacob(6, 15) = jacob(15, 6);
	jacob(6, 16) = jacob(16, 6);
	jacob(6, 17) = jacob(17, 6);
	jacob(6, 18) = jacob(18, 6);
	jacob(6, 19) = jacob(19, 6);
	jacob(6, 20) = jacob(20, 6);
	jacob(6, 21) = jacob(21, 6);
	jacob(6, 22) = jacob(22, 6);
	jacob(6, 23) = jacob(23, 6);
	jacob(7, 8) = jacob(8, 7);
	jacob(7, 9) = jacob(9, 7);
	jacob(7, 10) = jacob(10, 7);
	jacob(7, 11) = jacob(8, 10);
	jacob(7, 11) = jacob(10, 8);
	jacob(7, 11) = jacob(11, 7);
	jacob(7, 12) = jacob(12, 7);
	jacob(7, 13) = jacob(13, 7);
	jacob(7, 14) = jacob(8, 13);
	jacob(7, 14) = jacob(13, 8);
	jacob(7, 14) = jacob(14, 7);
	jacob(7, 15) = jacob(15, 7);
	jacob(7, 16) = jacob(16, 7);
	jacob(7, 17) = jacob(8, 16);
	jacob(7, 17) = jacob(16, 8);
	jacob(7, 17) = jacob(17, 7);
	jacob(7, 18) = jacob(18, 7);
	jacob(7, 19) = jacob(19, 7);
	jacob(7, 20) = jacob(8, 19);
	jacob(7, 20) = jacob(19, 8);
	jacob(7, 20) = jacob(20, 7);
	jacob(7, 21) = jacob(21, 7);
	jacob(7, 22) = jacob(22, 7);
	jacob(7, 23) = jacob(8, 22);
	jacob(7, 23) = jacob(22, 8);
	jacob(7, 23) = jacob(23, 7);
	jacob(8, 9) = jacob(9, 8);
	jacob(8, 10) = jacob(10, 8);
	jacob(8, 10) = jacob(11, 7);
	jacob(8, 11) = jacob(11, 8);
	jacob(8, 12) = jacob(12, 8);
	jacob(8, 13) = jacob(13, 8);
	jacob(8, 13) = jacob(14, 7);
	jacob(8, 14) = jacob(14, 8);
	jacob(8, 15) = jacob(15, 8);
	jacob(8, 16) = jacob(16, 8);
	jacob(8, 16) = jacob(17, 7);
	jacob(8, 17) = jacob(17, 8);
	jacob(8, 18) = jacob(18, 8);
	jacob(8, 19) = jacob(19, 8);
	jacob(8, 19) = jacob(20, 7);
	jacob(8, 20) = jacob(20, 8);
	jacob(8, 21) = jacob(21, 8);
	jacob(8, 22) = jacob(22, 8);
	jacob(8, 22) = jacob(23, 7);
	jacob(8, 23) = jacob(23, 8);
	jacob(9, 10) = jacob(10, 9);
	jacob(9, 11) = jacob(11, 9);
	jacob(9, 12) = jacob(12, 9);
	jacob(9, 13) = jacob(13, 9);
	jacob(9, 14) = jacob(14, 9);
	jacob(9, 15) = jacob(15, 9);
	jacob(9, 16) = jacob(16, 9);
	jacob(9, 17) = jacob(17, 9);
	jacob(9, 18) = jacob(18, 9);
	jacob(9, 19) = jacob(19, 9);
	jacob(9, 20) = jacob(20, 9);
	jacob(9, 21) = jacob(21, 9);
	jacob(9, 22) = jacob(22, 9);
	jacob(9, 23) = jacob(23, 9);
	jacob(10, 8) = jacob(11, 7);
	jacob(10, 11) = jacob(11, 10);
	jacob(10, 12) = jacob(12, 10);
	jacob(10, 13) = jacob(13, 10);
	jacob(10, 14) = jacob(11, 13);
	jacob(10, 14) = jacob(13, 11);
	jacob(10, 14) = jacob(14, 10);
	jacob(10, 15) = jacob(15, 10);
	jacob(10, 16) = jacob(16, 10);
	jacob(10, 17) = jacob(11, 16);
	jacob(10, 17) = jacob(16, 11);
	jacob(10, 17) = jacob(17, 10);
	jacob(10, 18) = jacob(18, 10);
	jacob(10, 19) = jacob(19, 10);
	jacob(10, 20) = jacob(11, 19);
	jacob(10, 20) = jacob(19, 11);
	jacob(10, 20) = jacob(20, 10);
	jacob(10, 21) = jacob(21, 10);
	jacob(10, 22) = jacob(22, 10);
	jacob(10, 23) = jacob(11, 22);
	jacob(10, 23) = jacob(22, 11);
	jacob(10, 23) = jacob(23, 10);
	jacob(11, 12) = jacob(12, 11);
	jacob(11, 13) = jacob(13, 11);
	jacob(11, 13) = jacob(14, 10);
	jacob(11, 14) = jacob(14, 11);
	jacob(11, 15) = jacob(15, 11);
	jacob(11, 16) = jacob(16, 11);
	jacob(11, 16) = jacob(17, 10);
	jacob(11, 17) = jacob(17, 11);
	jacob(11, 18) = jacob(18, 11);
	jacob(11, 19) = jacob(19, 11);
	jacob(11, 19) = jacob(20, 10);
	jacob(11, 20) = jacob(20, 11);
	jacob(11, 21) = jacob(21, 11);
	jacob(11, 22) = jacob(22, 11);
	jacob(11, 22) = jacob(23, 10);
	jacob(11, 23) = jacob(23, 11);
	jacob(12, 13) = jacob(13, 12);
	jacob(12, 14) = jacob(14, 12);
	jacob(12, 15) = jacob(15, 12);
	jacob(12, 16) = jacob(16, 12);
	jacob(12, 17) = jacob(17, 12);
	jacob(12, 18) = jacob(18, 12);
	jacob(12, 19) = jacob(19, 12);
	jacob(12, 20) = jacob(20, 12);
	jacob(12, 21) = jacob(21, 12);
	jacob(12, 22) = jacob(22, 12);
	jacob(12, 23) = jacob(23, 12);
	jacob(13, 8) = jacob(14, 7);
	jacob(13, 11) = jacob(14, 10);
	jacob(13, 14) = jacob(14, 13);
	jacob(13, 15) = jacob(15, 13);
	jacob(13, 16) = jacob(16, 13);
	jacob(13, 17) = jacob(14, 16);
	jacob(13, 17) = jacob(16, 14);
	jacob(13, 17) = jacob(17, 13);
	jacob(13, 18) = jacob(18, 13);
	jacob(13, 19) = jacob(19, 13);
	jacob(13, 20) = jacob(14, 19);
	jacob(13, 20) = jacob(19, 14);
	jacob(13, 20) = jacob(20, 13);
	jacob(13, 21) = jacob(21, 13);
	jacob(13, 22) = jacob(22, 13);
	jacob(13, 23) = jacob(14, 22);
	jacob(13, 23) = jacob(22, 14);
	jacob(13, 23) = jacob(23, 13);
	jacob(14, 15) = jacob(15, 14);
	jacob(14, 16) = jacob(16, 14);
	jacob(14, 16) = jacob(17, 13);
	jacob(14, 17) = jacob(17, 14);
	jacob(14, 18) = jacob(18, 14);
	jacob(14, 19) = jacob(19, 14);
	jacob(14, 19) = jacob(20, 13);
	jacob(14, 20) = jacob(20, 14);
	jacob(14, 21) = jacob(21, 14);
	jacob(14, 22) = jacob(22, 14);
	jacob(14, 22) = jacob(23, 13);
	jacob(14, 23) = jacob(23, 14);
	jacob(15, 16) = jacob(16, 15);
	jacob(15, 17) = jacob(17, 15);
	jacob(15, 18) = jacob(18, 15);
	jacob(15, 19) = jacob(19, 15);
	jacob(15, 20) = jacob(20, 15);
	jacob(15, 21) = jacob(21, 15);
	jacob(15, 22) = jacob(22, 15);
	jacob(15, 23) = jacob(23, 15);
	jacob(16, 8) = jacob(17, 7);
	jacob(16, 11) = jacob(17, 10);
	jacob(16, 14) = jacob(17, 13);
	jacob(16, 17) = jacob(17, 16);
	jacob(16, 18) = jacob(18, 16);
	jacob(16, 19) = jacob(19, 16);
	jacob(16, 20) = jacob(17, 19);
	jacob(16, 20) = jacob(19, 17);
	jacob(16, 20) = jacob(20, 16);
	jacob(16, 21) = jacob(21, 16);
	jacob(16, 22) = jacob(22, 16);
	jacob(16, 23) = jacob(17, 22);
	jacob(16, 23) = jacob(22, 17);
	jacob(16, 23) = jacob(23, 16);
	jacob(17, 18) = jacob(18, 17);
	jacob(17, 19) = jacob(19, 17);
	jacob(17, 19) = jacob(20, 16);
	jacob(17, 20) = jacob(20, 17);
	jacob(17, 21) = jacob(21, 17);
	jacob(17, 22) = jacob(22, 17);
	jacob(17, 22) = jacob(23, 16);
	jacob(17, 23) = jacob(23, 17);
	jacob(18, 19) = jacob(19, 18);
	jacob(18, 20) = jacob(20, 18);
	jacob(18, 21) = jacob(21, 18);
	jacob(18, 22) = jacob(22, 18);
	jacob(18, 23) = jacob(23, 18);
	jacob(19, 8) = jacob(20, 7);
	jacob(19, 11) = jacob(20, 10);
	jacob(19, 14) = jacob(20, 13);
	jacob(19, 17) = jacob(20, 16);
	jacob(19, 20) = jacob(20, 19);
	jacob(19, 21) = jacob(21, 19);
	jacob(19, 22) = jacob(22, 19);
	jacob(19, 23) = jacob(20, 22);
	jacob(19, 23) = jacob(22, 20);
	jacob(19, 23) = jacob(23, 19);
	jacob(20, 21) = jacob(21, 20);
	jacob(20, 22) = jacob(22, 20);
	jacob(20, 22) = jacob(23, 19);
	jacob(20, 23) = jacob(23, 20);
	jacob(21, 22) = jacob(22, 21);
	jacob(21, 23) = jacob(23, 21);
	jacob(22, 8) = jacob(23, 7);
	jacob(22, 11) = jacob(23, 10);
	jacob(22, 14) = jacob(23, 13);
	jacob(22, 17) = jacob(23, 16);
	jacob(22, 20) = jacob(23, 19);
	jacob(22, 23) = jacob(23, 22);

	for(int i=0;i<24;i++)
		jacobianDiagonals[i] = jacob(i,i);

	while(1){

		for(int i=0;i<24;i++){
			jacob(i, i) = jacobianDiagonals[i] * (lambda + 1);
		}

		getUpdate(jacob, fVector, 24, updates);


		for(int i=0;i<24;i++){
			newParams[i] = params[i] - updates[i];
		}

		float newCost = FullModelCostWelsch(corners1, corners2, w, newParams);
		//printf("newCost: %f  lambda %f\n", newCost, lambda);
		if(isnan(newCost)){
			printf("new cost is NAN\n");
			for(int i=0;i<24;i++){
				printf("old%f  new: %f\n", params[i], newParams[i]);
				updates[i] = 0;
			}
			lambda *= LAMBDA_INCREASE;	//maybe have a better change next time?
			break;
		}
		else if(newCost > startCost){
			lambda *= LAMBDA_INCREASE;
			if(lambda > MAX_LAMBDA)
			{
				for(int i=0;i<24;i++){
					updates[i] = 0;
					break;
				}
			}
		} else {
			lambda /= LAMBDA_DECREASE;
			break;
		}

	}
}

float JelloComplex2::FullModelCostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float w, float* params){
	float result = 0;
	for(int i=0;i<(int)corners1.size();i++){
		float x, y, x2, y2;
		x = corners1[i].x;
		y = corners1[i].y;
		x2 = corners2[i].x;
		y2 = corners2[i].y;

		EXTRACT_PARAMS(params)
		float x2Pred = dx0 + dx1*y + dx2_0*sin(dx3_0*y + dx4_0) + dx2_1*sin(dx3_1*y + dx4_1) + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
		float y2Pred = dy0 + dy1*y + dy2_0*sin(dy3_0*y + dy4_0) + dy2_1*sin(dy3_1*y + dy4_1) + x*sin(r0 + r1*y) + y*cos(r0 + r1*y) + r2_0*(-centerX + x)*sin(r3_0*y + r4_0)/(2*centerX) + r2_1*(-centerX + x)*sin(r3_1*y + r4_1)/(2*centerX);
		
		float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
		result += 1.0 - exp(-d/(w*w));
	}	

	result *= w*w/(float)corners1.size();

	return result;
}

float JelloComplex2::FullModelCostWelschXY(vector<Point2f> corners1, vector<Point2f> corners2, float w, float* params, float &ex, float &ey){
	float result = 0;
	float resultX = 0, resultY = 0;
	for(int i=0;i<(int)corners1.size();i++){
		float x, y, x2, y2;
		x = corners1[i].x;
		y = corners1[i].y;
		x2 = corners2[i].x;
		y2 = corners2[i].y;

		EXTRACT_PARAMS(params)
		float x2Pred = dx0 + dx1*y + dx2_0*sin(dx3_0*y + dx4_0) + dx2_1*sin(dx3_1*y + dx4_1) + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
		float y2Pred = dy0 + dy1*y + dy2_0*sin(dy3_0*y + dy4_0) + dy2_1*sin(dy3_1*y + dy4_1) + x*sin(r0 + r1*y) + y*cos(r0 + r1*y) + r2_0*(-centerX + x)*sin(r3_0*y + r4_0)/(2*centerX) + r2_1*(-centerX + x)*sin(r3_1*y + r4_1)/(2*centerX);
		
		float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
		float dx = (x2-x2Pred)*(x2-x2Pred);
		float dy = (y2-y2Pred)*(y2-y2Pred);

		result += 1.0 - exp(-d/(w*w));
		resultX += 1.0 - exp(-dx/(w*w));
		resultY += 1.0 - exp(-dy/(w*w));
	}	

	result *= w*w/(float)corners1.size();
	resultX *= w*w/(float)corners1.size();
	resultY *= w*w/(float)corners1.size();

	ex = resultX;
	ey = resultY;

	return result;
}

float JelloComplex2::FullModelCostLs(vector<Point2f> corners1, vector<Point2f> corners2, float* params){
	float result = 0;
	for(int i=0;i<(int)corners1.size();i++){
		float x, y, x2, y2;
		x = corners1[i].x;
		y = corners1[i].y;
		x2 = corners2[i].x;
		y2 = corners2[i].y;

		EXTRACT_PARAMS(params)
		float x2Pred = dx0 + dx1*y + dx2_0*sin(dx3_0*y + dx4_0) + dx2_1*sin(dx3_1*y + dx4_1) + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
		float y2Pred = dy0 + dy1*y + dy2_0*sin(dy3_0*y + dy4_0) + dy2_1*sin(dy3_1*y + dy4_1) + x*sin(r0 + r1*y) + y*cos(r0 + r1*y) + r2_0*(-centerX + x)*sin(r3_0*y + r4_0)/(2*centerX) + r2_1*(-centerX + x)*sin(r3_1*y + r4_1)/(2*centerX);
		
		float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
		result += d;
	}	

	result /= (float)corners1.size();
	result = sqrt(result);

	return result;
}

void JelloComplex2::GetSineEstimatesWeighted2(vector<float> ys, vector<float> diffs, vector<float> weights, vector<float> startingWeights, float* &result){

	float bestTotal = 0;
	float bestFreq, bestSin, bestCos;
	
	for(int i=0;i<F_STEPS;i++){
		float freq = F_LOW + (F_HIGH-F_LOW) * i / (F_STEPS-1);
		float sinSum = 0, cosSum = 0;
		for(int j=0;j<ys.size();j++){
			sinSum += sin(ys[j] * freq) * diffs[j] * weights[j] * startingWeights[j];
			cosSum += cos(ys[j] * freq) * diffs[j] * weights[j] * startingWeights[j];
		}

		float total = sinSum * sinSum + cosSum * cosSum;
	
		if(total > bestTotal){
			bestTotal = total;
			bestFreq = freq;
			bestSin = sinSum;
			bestCos = cosSum;
		}
	}

	float weightsSquaredSum = 0;
	for(int i=0;i<weights.size();i++) weightsSquaredSum += weights[i] * weights[i];

	float amplitude = 2 * sqrt(bestTotal) / weightsSquaredSum;
	float phase = atan2(bestCos, bestSin);

	result[0] = amplitude;
	result[1] = bestFreq;
	result[2] = phase;

	//printf("starting values: %f %f %f\n", amplitude, bestFreq, phase);

	//do some LM minimization here
	float lambda = SIN_MODEL_LAMBDA;
	float* updates = new float[3];
	float startW = SIN_MODEL_START_W;
	float endW = SIN_MODEL_END_W;
	float steps = SIN_MODEL_STEPS;
	for(int i=0;i<steps;i++){
		float w = pow(10,  log10(startW) + (log10(endW) - log10(startW)) * (i / (steps-1)));
		//w = 10;
		ImproveSineEstimatesWelsch(ys, diffs, weights, (int)ys.size(), result, updates, lambda, w);
		for(int j=0;j<3;j++){
			result[j] -= updates[j];
			//printf("%f\t", result[j]);
		}
		//float cost = SineEstimatesCostWelsch(ys, diffs, weights, result, w);
		//printf("w: %f  lambda: %f    cost: %f\n", w, lambda, cost);
		//exit(0);
	}
}

float JelloComplex2::SineEstimatesCostWelsch(vector<float> ys, vector<float> diffs, vector<float> weights, float* params, float w){
	float result = 0;

	float amplitude = params[0];
	float freq = params[1];
	float phase = params[2];

	for(int i=0;i<(int)ys.size();i++){
		float pred = amplitude * sin(ys[i] * freq + phase) * weights[i];
		float d = (pred-diffs[i]) * (pred-diffs[i]);
		result += w*w*(1-exp(-d/(w*w)));
	}

	result /= (float)(ys.size());
	return result;
}

#define JACOBIAN_MIN_DIAGONAL	(1e-6)
void JelloComplex2::getUpdate(arma::Mat<float> jacob, arma::Col<float> fVector, int numParams, float* &updates){

	arma::Col<FLOAT_TYPE> update;

	for(int i=0;i<numParams;i++){
		if(fabs(jacob(i,i)) < JACOBIAN_MIN_DIAGONAL){

			//printf("parameter %d is too small\n", i);
			//jacob.print();

			for(int j=0;j<numParams;j++){
				jacob(i,j) = 0;
				jacob(j,i) = 0;
			}

			fVector(i) = 0;
			jacob(i, i) = 1;

			//jacob.print();
		}
	}

	try{
		update = arma::solve(jacob, fVector);
	} catch (const runtime_error& error){
	   	printf("solve() threw a runtime error\n");
	   	printf("jacobian:\n");
	   	jacob.print();
	   	printf("fVect: \n");
	   	fVector.print();
		printf("length of inputs: %d\n", numParams);
	   	exit(0);
	   }

	   for(int i=0;i<numParams;i++){
	   	updates[i] = update(i);
	   }
}

void JelloComplex2::ImproveSineEstimatesWelsch(vector<float> ys, vector<float> diffs, vector<float> weights, int length, float* params, float* &updates, float &lambda, float w){

	arma::Col<FLOAT_TYPE> fVector(3);
	arma::Mat<FLOAT_TYPE> jacob(3,3);
	fVector.zeros();
	jacob.zeros();

	FLOAT_TYPE a = params[0];
	FLOAT_TYPE b = params[1];
	FLOAT_TYPE c = params[2];

	float newParams[3];
	float jacobianDiagonals[3];

	float startCost = SineEstimatesCostWelsch(ys, diffs, weights, params, w);

	for(int i=0;i<length;i++){
		float y = ys[i];
		float cw = weights[i];

		float d = diffs[i];

		if(cw < 1e-5){
			continue;
		}

		FLOAT_TYPE t0 =  b*y;
		FLOAT_TYPE t1 =  c + t0;
		FLOAT_TYPE t2 =  sin(t1);
		FLOAT_TYPE t3 =  a*t2;
		FLOAT_TYPE t4 =  cw*t3;
		FLOAT_TYPE t5 =  d - t4;
		FLOAT_TYPE t6 =  (t5*t5);
		FLOAT_TYPE t7 =  pow(w, -2);
		FLOAT_TYPE t8 =  t6*t7;
		FLOAT_TYPE t9 =  exp(-t8);
		FLOAT_TYPE t10 =  cos(t1);
		FLOAT_TYPE t11 =  (cw*cw);
		FLOAT_TYPE t12 =  t11*t9;
		FLOAT_TYPE t13 =  cw*t9;
		FLOAT_TYPE t14 =  t13*t5;
		FLOAT_TYPE t15 =  -4*t12;
		FLOAT_TYPE t16 =  2*t12;
		FLOAT_TYPE t17 =  (t10*t10);
		FLOAT_TYPE t18 =  (a*a)*t17;
		FLOAT_TYPE t19 =  a*t10;
		FLOAT_TYPE t20 =  t15*t6*t7;
		FLOAT_TYPE t21 =  t19*t2;
		FLOAT_TYPE t22 =  t14*y;
		FLOAT_TYPE t23 =  t20*y;
		FLOAT_TYPE t24 =  -2*t10;
		FLOAT_TYPE t25 =  t16*y;

		jacob(0, 0) +=  (t2*t2)*(t16 + t20);
		jacob(1, 0) +=  t21*t23 + t21*t25 + t22*t24;
		jacob(1, 1) +=  (y*y)*(2*a*t14*t2 + t16*t18 + t18*t20);
		jacob(2, 0) +=  t10*t15*t3*t8 + t14*t24 + t16*t21;
		jacob(2, 1) +=  t18*t23 + t18*t25 + 2*t22*t3;
		jacob(2, 2) +=  2*a*t14*t2 + t16*t18 + t18*t20;

		fVector(0) +=  t14*t2;
		fVector(1) +=  t19*t22;
		fVector(2) +=  t14*t19;
	}
	
	fVector(0) *= -2.000000;
	fVector(1) *= -2.000000;
	fVector(2) *= -2.000000;


	jacob(0, 1) = jacob(1, 0);
	jacob(0, 2) = jacob(2, 0);
	jacob(1, 2) = jacob(2, 1);
	
	for(int i=0;i<3;i++)
		jacobianDiagonals[i] = jacob(i,i);


	//printf("\n"); printf("%f %f %f \n", a, b, c); jacob.print();
	//printf("%e %f %f   -  cost(4): %f\n", a, b, c, SineEstimatesCostWelsch(ys, diffs, weights, newParams, 4)); 
	while(1){

		for(int i=0;i<3;i++){
			jacob(i, i) = jacobianDiagonals[i] * (lambda + 1);
		}
		//printf("lambda: %f\n", lambda);

		getUpdate(jacob, fVector, 3, updates);
		
		for(int i=0;i<3;i++){
			newParams[i] = params[i] - updates[i];
		}

		float newCost = SineEstimatesCostWelsch(ys, diffs, weights, newParams, w);

		if(isnan(newCost)){
			printf("new cost is NAN in GetSineEstimatesWeighted()");
			printf("jacobian: \n");
			jacob.print();
			printf("f vector: \n");
			fVector.print();
			printf("input params: %f %f %f\n", params[0], params[1], params[2]);
			exit(0);
			lambda *= LAMBDA_INCREASE;
			updates[0] = 0;
			updates[1] = 0;
			updates[2] = 0;
			break;
		} else if(newCost > startCost){
			lambda *= LAMBDA_INCREASE;
			if(lambda > MAX_LAMBDA){
				updates[0] = 0;
				updates[1] = 0;
				updates[2] = 0;
				break;
			}
		} else {
			lambda /= LAMBDA_DECREASE;
			break;
		}

	}
}

float JelloComplex2::getModel2CostWelsch(vector<Point2f> corners1, vector<Point2f> corners2, float* params, float w){
	float result = 0;
	for(int i=0;i<(int)corners1.size();i++){
		float x, y, x2, y2;
		x = corners1[i].x;
		y = corners1[i].y;
		x2 = corners2[i].x;
		y2 = corners2[i].y;

		EXTRACT_PARAMS(params)
		float x2Pred = dx0 + dx1*y + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
		float y2Pred = dy0 + dy1*y + x*sin(r0 + r1*y) + y*cos(r0 + r1*y);
		
		float d = (x2-x2Pred) * (x2-x2Pred) + (y2 - y2Pred) * (y2-y2Pred);
		float e = w*w*(1.0 - exp(-d/(w*w)));
		result += e;
	}	

	result /= (float)corners1.size();
	result = sqrt(result);

	return result;
}

void JelloComplex2::model2LMIterationWelsch(vector<Point2f> corners1, vector<Point2f> corners2, int length, float* params, float* &updates, float &lambda, float w){

	arma::Col<FLOAT_TYPE> fVector(6);
	arma::Mat<FLOAT_TYPE> jacob(6,6);
	arma::Col<FLOAT_TYPE> update;
	fVector.zeros();
	jacob.zeros();

	FLOAT_TYPE r0 = params[0];
	FLOAT_TYPE dx0 = params[1];
	FLOAT_TYPE dy0 = params[2];
	FLOAT_TYPE r1 = params[3];
	FLOAT_TYPE dx1 = params[4];
	FLOAT_TYPE dy1 = params[5];

	float newParams[6];
	float jacobianDiagonals[6];

	float startCost = getModel2CostWelsch(corners1, corners2, params, w);

	for(int i=0;i<length;i++){
		FLOAT_TYPE x1 = corners1[i].x;
		FLOAT_TYPE y1 = corners1[i].y;
		FLOAT_TYPE x2 = corners2[i].x;
		FLOAT_TYPE y2 = corners2[i].y;

		FLOAT_TYPE t0 =  r0 + r1*y1;
		FLOAT_TYPE t1 =  cos(t0);
		FLOAT_TYPE t2 =  sin(t0);
		FLOAT_TYPE t3 =  t2*y1;
		FLOAT_TYPE t4 =  -t3;
		FLOAT_TYPE t5 =  t1*x1;
		FLOAT_TYPE t6 =  t1*y1;
		FLOAT_TYPE t7 =  dy0 + dy1*y1 + t2*x1 + t6 - y2;
		FLOAT_TYPE t8 =  dx0 + dx1*y1 + t4 + t5 - x2;
		FLOAT_TYPE t9 =  pow(w, -2);
		FLOAT_TYPE t10 =  (t7*t7);
		FLOAT_TYPE t11 =  (t8*t8);
		FLOAT_TYPE t12 =  -t11;
		FLOAT_TYPE t13 =  -t10;
		FLOAT_TYPE t14 =  t12 + t13;
		FLOAT_TYPE t15 =  exp(t14*t9);
		FLOAT_TYPE t16 =  -2*t1;
		FLOAT_TYPE t17 =  (y1*y1);
		FLOAT_TYPE t18 =  -2*x1;
		FLOAT_TYPE t19 =  2*y1;
		FLOAT_TYPE t20 =  t15*t9;
		FLOAT_TYPE t21 =  t1*x1;
		FLOAT_TYPE t22 =  t18*t2;
		FLOAT_TYPE t23 =  t16*y1;
		FLOAT_TYPE t24 =  -t7;
		FLOAT_TYPE t25 =  -t8;
		FLOAT_TYPE t26 =  -2*y1;
		FLOAT_TYPE t27 =  t17*t2;
		FLOAT_TYPE t28 =  t16*t17;
		FLOAT_TYPE t29 =  t18*t3 + t28;
		FLOAT_TYPE t30 =  t19*t20;
		FLOAT_TYPE t31 =  t22 + t23;
		FLOAT_TYPE t32 =  2*t21 - 2*t3;
		FLOAT_TYPE t33 =  t19*t21 - 2*t27;
		FLOAT_TYPE t34 =  2*t3;
		FLOAT_TYPE t35 =  t16*x1;
		FLOAT_TYPE t36 =  -t20;
		FLOAT_TYPE t37 =  t24*t32 + t25*t31;
		FLOAT_TYPE t38 =  2*y2;
		FLOAT_TYPE t39 =  -2*dx0 + dx1*t26 + t34 + t35 + 2*x2;
		FLOAT_TYPE t40 =  -t15;
		FLOAT_TYPE t41 =  -2*dy0 + dy1*t26 + t22 + t23 + t38;
		FLOAT_TYPE t42 =  t25*t29;
		FLOAT_TYPE t43 =  t24*t33 + t42;
		FLOAT_TYPE t44 =  t30*t7;
		FLOAT_TYPE t45 =  t30*t8;
		FLOAT_TYPE t46 =  t36*t41;
		FLOAT_TYPE t47 =  -t1;
		FLOAT_TYPE t48 =  -x1;
		FLOAT_TYPE t49 =  t15*t19;
		FLOAT_TYPE t50 =  t16*x1;
		FLOAT_TYPE t51 =  t17*t20;
		FLOAT_TYPE t52 =  2*t27;
		FLOAT_TYPE t53 =  2*t17;
		FLOAT_TYPE t54 =  t50*y1;
		FLOAT_TYPE t55 =  t36*t39;
		FLOAT_TYPE t56 =  t15*y1;
		FLOAT_TYPE t57 =  t52 + t54;
		FLOAT_TYPE t58 =  2*x1;

		jacob(0, 0) +=  t36*(t37*t37) + t40*(t24*t31 + t25*(t34 + t35) - t31*(t2*t48 + t47*y1) - t32*(t4 + t5));
		jacob(0, 3) +=  t36*t37*t43 + t40*(t24*t29 + t25*t57 - t31*(t17*t47 + t3*t48) - t32*(t21*y1 - t27));
		jacob(0, 4) +=  t31*t56 + t37*t45;
		jacob(0, 5) +=  t32*t56 + t37*t44;
		jacob(1, 0) +=  t37*t55 + t40*(t1*t19 + t2*t58);
		jacob(1, 1) +=  2*t15 + t36*(t39*t39);
		jacob(2, 0) +=  t37*t46 + t40*(t34 + t35);
		jacob(2, 1) +=  t39*t46;
		jacob(2, 2) +=  2*t15 + t36*(t41*t41);
		jacob(3, 0) +=  t36*t37*t43 + t40*(t24*t29 + t25*t57 - t29*(t2*t48 + t47*y1) - t33*(t4 + t5));
		jacob(3, 1) +=  t40*(t1*t53 + t3*t58) + t43*t55;
		jacob(3, 2) +=  t40*t57 + t43*t46;
		jacob(3, 3) +=  t36*(t43*t43) + t40*(t24*(t16*pow(y1, 3) + t18*t27) + t25*(t17*t50 + 2*t2*pow(y1, 3)) - t29*(t17*t47 + t3*t48) - t33*(t21*y1 - t27));
		jacob(3, 4) +=  t29*t56 + t43*t45;
		jacob(3, 5) +=  t33*t56 + t43*t44;
		jacob(4, 0) +=  t37*t45 + t49*(t2*t48 + t47*y1);
		jacob(4, 1) +=  t39*t45 + t49;
		jacob(4, 2) +=  t41*t45;
		jacob(4, 3) +=  t43*t45 + t49*(t17*t47 + t3*t48);
		jacob(4, 4) +=  -4*t11*t51 + t15*t53;
		jacob(5, 0) +=  t37*t44 + t49*(t4 + t5);
		jacob(5, 1) +=  t39*t44;
		jacob(5, 2) +=  t41*t44 + t49;
		jacob(5, 3) +=  t43*t44 + t49*(t21*y1 - t27);
		jacob(5, 4) +=  t51*t7*t8;
		jacob(5, 5) +=  -4*t10*t51 + t15*t53;

		fVector(0) +=  t37*t40;
		fVector(1) +=  t39*t40;
		fVector(2) +=  t40*t41;
		fVector(3) +=  t40*t43;
		fVector(4) +=  t49*t8;
		fVector(5) +=  t49*t7;
	}

	jacob(5, 4) *= -4.000000;


	jacob(0, 1) = jacob(1, 0);
	jacob(0, 2) = jacob(2, 0);
	jacob(1, 2) = jacob(2, 1);
	jacob(1, 3) = jacob(3, 1);
	jacob(1, 4) = jacob(4, 1);
	jacob(1, 5) = jacob(5, 1);
	jacob(2, 3) = jacob(3, 2);
	jacob(2, 4) = jacob(4, 2);
	jacob(2, 5) = jacob(5, 2);
	jacob(4, 5) = jacob(5, 4);


	for(int i=0;i<6;i++)
		jacobianDiagonals[i] = jacob(i,i);

	while(1){

		for(int i=0;i<6;i++){
			jacob(i, i) = jacobianDiagonals[i] * (lambda + 1);
		}
		//printf("lambda: %f\n", lambda);
		update = jacob.i() * fVector;

		for(int i=0;i<6;i++){
			newParams[i] = params[i] - update(i);
		}

		float newCost = getModel2CostWelsch(corners1, corners2, newParams, w);
		//printf("startcost: %f   newCost: %f\n", startCost, newCost);

		if(isnan(newCost)){
			printf("new cost is NAN in model2LMIterationWelsch()");
			lambda *= LAMBDA_INCREASE;
			update(0) = 0;
			update(1) = 0;
			update(2) = 0;
			break;
		} else if(newCost > startCost){
			lambda *= LAMBDA_INCREASE;
			if(lambda > MAX_LAMBDA){
				update(0) = 0;
				update(1) = 0;
				update(2) = 0;
				break;
			}
		} else {
			lambda /= LAMBDA_DECREASE;
			break;
		}

	}

	for(int i=0;i<6;i++){
		updates[i] = update(i);
	}
}

void JelloComplex2::CalcJelloTransform(Mat img1, Mat img2){
	params = new float[NUM_PARAMS];
	for(int i=0;i<NUM_PARAMS;i++){
		params[i] = 0;
	}

	vector<Point2f> corners1, corners2;
	int length = GetPointsToTrack(img1, img2, corners1, corners2);
	

	timeval start;
	gettimeofday(&start, NULL);
	long startMs = (start.tv_sec * 1000) + (start.tv_usec/1000);
	
	CalculateModelParameters(corners1, corners2, length, params);

	timeval end;
	gettimeofday(&end, NULL);
	long endMs = (end.tv_sec * 1000) + (end.tv_usec/1000);
	long elapsedMs = endMs - startMs;
	//msTotal += elapsedMs;
	//printf("elapsed ms: %d\n", (int)elapsedMs);
}

void JelloComplex2::CreateAbsoluteTransform(JelloComplex2 prevTransform){
	for(int row=0;row<imgHeight;row++){
		for(int col=0;col<imgWidth;col++){
			float x2, y2;
			
			float x = col + prevTransform.shiftsX[row][col];
			float y = row + prevTransform.shiftsY[row][col];
			TransformPoint(x, y, x2, y2);

			shiftsX[row][col] = prevTransform.shiftsX[row][col] * JELLO_DECAY - x2 + x;
			shiftsY[row][col] = prevTransform.shiftsY[row][col] * JELLO_DECAY - y2 + y;
		}
	}
}

void JelloComplex2::TransformPoint(float x, float y, float &x2, float &y2){
	EXTRACT_PARAMS(params)
	x2 = dx0 + dx1*y + dx2_0*sin(dx3_0*y + dx4_0) + dx2_1*sin(dx3_1*y + dx4_1) + x*cos(r0 + r1*y) - y*sin(r0 + r1*y);
	y2 = dy0 + dy1*y + dy2_0*sin(dy3_0*y + dy4_0) + dy2_1*sin(dy3_1*y + dy4_1) + x*sin(r0 + r1*y) + y*cos(r0 + r1*y) + r2_0*(-centerX + x)*sin(r3_0*y + r4_0)/(2*centerX) + r2_1*(-centerX + x)*sin(r3_1*y + r4_1)/(2*centerX);
}

void JelloComplex2::TransformPointAbs(float x, float y, float &x2, float &y2){
	int ix = round(x);
	int iy = round(y);

	x2 = x - shiftsX[iy][ix];
	y2 = y - shiftsY[iy][ix];	
}	

void JelloComplex2::analyzeTransformAccuracies(){
	if(allWelschCosts.size() == 0){
		printf("no shifts were analyzed during the run\n");
		return;
	}

	float _mean = std::accumulate(frameErrors.begin(), frameErrors.end(), 0.0) / (float)frameErrors.size();
	printf("mean error: %f\n", _mean);

	float mean = std::accumulate(allWelschCosts.begin(), allWelschCosts.end(), 0.0) / (float)allWelschCosts.size();
	float meanStart = std::accumulate(initialWelschCosts.begin(), initialWelschCosts.end(), 0.0) / (float)initialWelschCosts.size();
	float fullFrameMean = std::accumulate(fullFrameWelschCosts.begin(), fullFrameWelschCosts.end(), 0.0) / (float)fullFrameWelschCosts.size();
	printf("initial cost: %f\n", meanStart);
	printf("full frame cost: %f\n", fullFrameMean);
	printf("mean welsch(4) cost: %f\n", mean);
}

