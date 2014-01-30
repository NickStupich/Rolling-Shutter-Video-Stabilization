#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "structures.h"
#include "svd.h"
#include "coreFuncs.h"

#include <math.h>
#include <set>
#include <armadillo>

using namespace std;
using namespace cv;

Transformation prunedWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights){
    Transformation t1 = weightedSvd(corners1, corners2, length, weights);

    int pruneCount = 0;
    int remainingPoints = 0;
    for(int i=0;i<length;i++)
    {
        float x = corners1[i].x;
        float y = corners1[i].y;
        
        float x2, y2;
        
        GenericTransformPoint(t1, x, y, x2, y2);
        float err = sqrt((x2-corners2[i].x) * (x2-corners2[i].x) + (y2-corners2[i].y) * (y2-corners2[i].y));
        //printf("meanCalc x: %f %f      y: %f %f      err: %f\n", x, corners1[i].x, y, corners1[i].y, err);
        
        if(err > SVD_PRUNE_MAX_DIST && weights[i] > 0)
        {
            weights[i] = 0;
            pruneCount++;
        } else if(weights[i] > 0){
            remainingPoints++;
        }
    }

    //printf("prune count: %d  remaining: %d\n", pruneCount, remainingPoints);

    Transformation t2 = weightedSvd(corners1, corners2, length, weights);
    return t2;
}

Transformation prunedNonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length){
    vector<float> weights(length);
    fill(weights.begin(), weights.end(), 1.0);
    
    return prunedWeightedSvd(corners1, corners2, length, weights);
}

Transformation weightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights){
    arma::fvec weightsArmaVec(length);
    
    float mean1x = 0., mean1y = 0., mean2x = 0., mean2y = 0., weightsSum = 0.;
                
    for(int i=0;i<length;i++)
    {
        mean1x += corners1[i].x * weights[i];
        mean1y += corners1[i].y * weights[i];
        mean2x += corners2[i].x * weights[i];
        mean2y += corners2[i].y * weights[i];
        
        weightsSum += weights[i];
        weightsArmaVec[i] = weights[i];
    }

    if(weightsSum == 0){
        printf("weights sum is 0\n");
        return (Transformation){0, 0, 0, 0, 0, 1, 0};
    }
    
    mean1x /= weightsSum;
    mean1y /= weightsSum;
    mean2x /= weightsSum;
    mean2y /= weightsSum;

    arma::fmat A(length, 2);
    arma::fmat B(length, 2);
    
    for(int i=0;i<length;i++)
    {
        A(i, 0) = corners1[i].x - mean1x;
        A(i, 1) = corners1[i].y - mean1y;
        B(i, 0) = corners2[i].x - mean2x;
        B(i, 1) = corners2[i].y - mean2y;
    }
    
    arma::fmat W = arma::diagmat(weightsArmaVec);
    arma::fmat H = A.t() * W * B;
    
    arma::fmat U;
    arma::fvec S;
    arma::fmat V;
    
    arma::svd(U, S, V, H);
    arma::fmat R = V.t() * U.t();
    
    arma::fvec centroidA(2);
    centroidA(0) = mean2x;
    centroidA(1) = mean2y;
    
    arma::fvec centroidB(2);
    centroidB(0) = mean1x;
    centroidB(1) = mean1y;
    
    arma::fvec t = centroidB - R*centroidA;
    
    Transformation trans;
    trans.rotation = asin(R(1,0));
    trans.cos = R(0,0);
    trans.sin = R(1,0);
    trans.ux1 = mean1x;
    trans.uy1 = mean1y;
    trans.ux2 = mean2x;
    trans.uy2 = mean2y;
    
    return trans;   
}

Transformation nonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length){
	vector<float> weights(length);
    fill(weights.begin(), weights.end(), 1.0);
    
    return weightedSvd(corners1, corners2, length, weights);
}

vector<float> getDensityWeights(vector<float> dxs, vector<float> dys, int length){
    int k = 20;
    
    vector<float> weights;
    
    for(int i=0;i<length;i++)
    {
        vector<float> distances;
        for(int j=0;j<length;j++)
        {
            if(i != j)
            {
                float dist = fabs(dxs[i] - dxs[j]) + fabs(dys[i] - dys[j]);
                distances.push_back(dist);
                
            }
        }
        
        std::sort(distances.begin(), distances.end());
            
        float totalDistance = 0.;
        for(int m=0;m<k;m++)
        {
            totalDistance += distances[m];
        }
        
        float weight = 1.0 / (totalDistance + 0.001);
        if(weight > 1000000){
            for(int j=0;j<50;j++)
            {
                printf("%f\n", distances[j]);

            }
            printf("totalDistance: %f\n", totalDistance);
            printf("weight: %f\n", weight);
            exit(0);
        }

        weights.push_back(weight);
    }

    return weights;
}

Transformation densityWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length){
    vector<float> dxs;
    vector<float> dys;
    
    for(int i=0;i<length;i++)
    {
        dxs.push_back(corners1[i].x - corners2[i].x);
        dys.push_back(corners1[i].y - corners2[i].y);
    }
    
    vector<float> weights = getDensityWeights(dxs, dys, length);

    Transformation result = weightedSvd(corners1, corners2, length, weights);

    return result;
}

set<int> randomPointSample(vector<Point2f> corners1, vector<Point2f> corners2, vector<Point2f> &sample1, vector<Point2f> &sample2, int inputLength, int sampleLength){

    sample1.clear();
    sample2.clear();

    set<int> indexes;

    if(sampleLength > inputLength){
        //printf("sample length (%d)  greater than population lengthm(%d), cannot draw sample\n", sampleLength, inputLength);
        return indexes;
    }

    while (indexes.size() < sampleLength)
    {
        int random_index = rand() % inputLength;
        if (indexes.find(random_index) == indexes.end())
        {
            sample1.push_back(corners1[random_index]);
            sample2.push_back(corners2[random_index]);

            indexes.insert(random_index);
        }
    }

    return indexes;
}

Transformation RansacNonWeightedSvd(vector<Point2f> corners1, vector<Point2f> corners2, int length){
    Transformation bestModel;
    int mostInliers = 0;
    int pointsPerIteration = 50;

    int maxIterations = 500;
    float inlierDistanceThreshold = 1.0;
    float inlierRatioThreshold = 0.99;
    int iteration;

    vector<Point2f> sample1;
    vector<Point2f> sample2;

    for(iteration=0;iteration<maxIterations;iteration++){

        set<int> indeces = randomPointSample(corners1, corners2, sample1, sample2, length, pointsPerIteration);
        if(indeces.size() == 0){
            return (Transformation){0, 0, 0, 0, 0, 1, 0};   //empty transformation
        }

        Transformation model = nonWeightedSvd(sample1, sample2, pointsPerIteration);

        for(int i=0;i<length;i++){
            if(indeces.find(i) == indeces.end()){
                float x2, y2;
                GenericTransformPoint(model, corners1[i].x, corners1[i].y, x2, y2);
                float d = sqrt(pow(corners2[i].x - x2, 2.0F) + pow(corners2[i].y - y2, 2.0F));
                if(d < inlierDistanceThreshold){
                    sample1.push_back(corners1[i]);
                    sample2.push_back(corners2[i]);
                }
            }
        }

        //printf("inliers: %d\n", (int)sample1.size());
        if(sample1.size() > mostInliers){
            bestModel = nonWeightedSvd(sample1, sample2, (int)sample1.size());
            mostInliers = sample1.size();

            if(sample1.size() >= inlierRatioThreshold * length){
                //printf("breaking after %d iterations, with %d points\n", iteration, (int)sample1.size());
                break;
            }
        }
    }
    printf("done RANSAC after %d iterations with %d / %d points as inliers\n", iteration, (int)sample1.size(), length);
    return bestModel;
}

//60 seconds / 10e6 iterations
void newtonIteration(float &t0, float &t1, float &r, float w,  vector<Point2f> corners1, vector<Point2f> corners2, int length){
    float f0 = 0, f1 = 0, f2 = 0;
    float j00 = 0, j01 = 0, j02 = 0, j10 = 0, j11 = 0, j12 = 0, j20 = 0, j21 = 0, j22 = 0;

    float cosr = cos(r);
    float sinr = sin(r);

    for(int i=0;i<length;i++){
        float x1 = corners1[i].x;
        float y1 = corners1[i].y;
        float x2 = corners2[i].x;
        float y2 = corners2[i].y;

        float s0 = pow(t0 + x1*cosr - x2 - y1*sinr,2.F);
        float s1 = (t1 + x1*sinr + y1*cosr - y2);
        float s2 = exp(-(s0 + s1*s1)/w)/(w*w);
        float s3 = exp((-s0 - s1*s1)/w)/w;
        float s4 = x1*sinr + y1*cosr;
        float s5 = t0 + x1*cosr - x2 - y1*sinr;
        float s6 = x1*cosr - y1*sinr;
        float s7 = exp(-(s0 + s1*s1)/w);
        float s8 = s4*s5 - s6*s1;
        float s9 = x1*cosr;
        float s10 = y1*cosr;
        float s11 = x1*sinr;
        float s12 = y1*sinr;
        float s13 = s7/w;

        f0 += 2.F*(-(-t0 - s9 + x2 + s12)*s3);
        f1 += 2.F*(-(-t1 - s11 - s10 + y2)*s3);
        f2 += 2.F*(-(-(-s11 - s10)*s5 - (s9 - s12)*s1)*s3);

        j00 += 2.F*((1 - 2.F*s0/w)*s13);
        j01 += -4*(s5*s1*s2);
        j02 += 2.F*((-s11 - s10 + 2.F*s8*s5/w)*s13);
        j10 += -4.F*(s5*s1*s2);
        j11 += 2.F*((1 - 2.F*s1*s1/w)*s13);
        j12 += 2.F*((s9 - s12 + 2.F*s8*s1/w)*s13);
        j20 +=2.F*((-s11 - s10 + 2.F*s8*s5/w)*s13);
        j21 += 2.F*((s9 - s12 + 2.F*s8*s1/w)*s13);
        j22 += 2.F*((s4*s4 - s4*s1 + s6*s6 - s6*s5 - 2.F*s8*s8/w)*s13);
    }

    float detInv = 1.0 / (j00 * (j22*j11 - j21*j12) - j10 * (j22*j01 - j21*j02) + j20 * (j12*j01-j11*j02));
    //printf("%f\n", detInv);
    float i00 = (j22*j11-j21*j12);
    float i01 = -(j22*j01 - j21*j02);
    float i02 = (j12*j01 - j11*j02);
    float i10 = -(j22*j10 - j20 * j12);
    float i11 = j22 *j00 - j20 * j02;
    float i12 = -(j12 * j00 - j10 * j02);
    float i20 = j21*j10 - j20 * j11;
    float i21 = -(j21*j00 - j20*j01);
    float i22 = j11*j00 - j10*j01 ;

    t0 -= detInv * (f0 * i00 + f1 * i01 + f2 * i02);
    t1 -= detInv * (f0 * i10 + f1 * i11 + f2 * i12);
    r -= detInv * (f0 * i20 + f1 * i21 + f2 * i22);
}

//49 seconds / 10e6 iterations
void newtonIteration2(float &t0, float &t1, float &r, float w,  vector<Point2f> corners1, vector<Point2f> corners2, int length){
    float f0 = 0.F, f1 = 0.F, f2 = 0.F;
    float j00 = 0.F, j01 = 0.F, j02 = 0.F, j11 = 0.F, j12 = 0.F, j22 = 0.F;

    float cosr = cos(r);
    float sinr = sin(r);
    float wInv = 1.0F / w;
    float wInv2 = wInv*wInv;

    float x1, y1, x2, y2;
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13;

    for(int i=0;i<length;i++){
        x1 = corners1[i].x;
        y1 = corners1[i].y;
        x2 = corners2[i].x;
        y2 = corners2[i].y;

        s9 = x1*cosr;
        s10 = y1*cosr;
        s11 = x1*sinr;
        s12 = y1*sinr;

        s0 = (t0 + s9- x2 - s12);
        s0 *= s0;
        s1 = t1 + s11 + s10 - y2;
        s7 = exp(-(s0 + s1*s1)*wInv);
        s2 = s7 * wInv2;
        s3 = exp((-s0 - s1*s1)*wInv)*wInv;
        s5 = t0 + s9 - x2 - s12;
        s8 = s4*s5 - s6*s1;
        s4 = s11 + s10;
        s6 = s9 - s12;
        s13 = s7*wInv;

        f0 += (-t0 - s9 + x2 + s12)*s3;
        f1 += (-t1 - s11 - s10 + y2)*s3;
        f2 += (-(-s11 - s10)*s5 - (s9 - s12)*s1)*s3;

        j00 += (1 - 2.F*s0*wInv)*s13;
        j01 -= s5*s1*s2;
        j02 += (-s11 - s10 + 2.F*s8*s5*wInv)*s13;
        j11 += (1 - 2.F*s1*s1*wInv)*s13;
        j12 += (s9 - s12 + 2.F*s8*s1*wInv)*s13;
        j22 += (s4*s4 - s4*s1 + s6*s6 - s6*s5 - 2.F*s8*s8*wInv)*s13;
    }

    j01 *= 2.F;

    float detInv = 1.0F / (j00 * (j22*j11 - j12*j12) - j01 * (j22*j01 - j12*j02) + j02 * (j12*j01-j11*j02));

    float i00 = (j22*j11-j12*j12);
    float i01 = -(j22*j01 - j12*j02);
    float i02 = (j12*j01 - j11*j02);
    float i10 = -(j22*j01 - j02 * j12);
    float i11 = j22 *j00 - j02 * j02;
    float i12 = -(j12 * j00 - j01 * j02);
    float i20 = j12*j01 - j02 * j11;
    float i21 = -(j12*j00 - j02*j01);
    float i22 = j11*j00 - j01*j01 ;

    t0 += detInv * (f0 * i00 + f1 * i01 + f2 * i02);
    t1 += detInv * (f0 * i10 + f1 * i11 + f2 * i12);
    r += detInv * (f0 * i20 + f1 * i21 + f2 * i22);
}

void newtonIteration2Weighted(float &t0, float &t1, float &r, float w,  vector<Point2f> corners1, vector<Point2f> corners2, vector<float> weights, int length){
    float f0 = 0.F, f1 = 0.F, f2 = 0.F;
    float j00 = 0.F, j01 = 0.F, j02 = 0.F, j11 = 0.F, j12 = 0.F, j22 = 0.F;

    float cosr = cos(r);
    float sinr = sin(r);
    float wInv = 1.0F / w;
    float wInv2 = wInv*wInv;

    float x1, y1, x2, y2, weight;
    float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13;

    //printf("j22 start: %f\n", j22);

    for(int i=0;i<length;i++){
        x1 = corners1[i].x;
        y1 = corners1[i].y;
        x2 = corners2[i].x;
        y2 = corners2[i].y;
        weight = weights[i];

        s9 = x1*cosr;
        s10 = y1*cosr;
        s11 = x1*sinr;
        s12 = y1*sinr;

        s0 = (t0 + s9- x2 - s12);
        s0 *= s0;
        s1 = t1 + s11 + s10 - y2;
        s7 = exp(-(s0 + s1*s1)*wInv);
        s2 = s7 * wInv2;
        s3 = exp((-s0 - s1*s1)*wInv)*wInv;
        s5 = t0 + s9 - x2 - s12;
        s8 = s4*s5 - s6*s1;
        s4 = s11 + s10;
        s6 = s9 - s12;
        s13 = s7*wInv;
        /*
        //printf("s: %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13);
        printf("s7: %f   j22 diff: %f\n", s7, weight * (s4*s4 - s4*s1 + s6*s6 - s6*s5 - 2.F*s8*s8*wInv)*s13); 
        printf("s13: %f   weight: %f\n", s13, weight);
        printf("%f %f %f %f %f\n", s4*s4, s4*s1, s6*s6, s6*s5, 2.F*s8*s8*wInv);
        printf("s8: %f\n", s8);

        if(s8 > 10e15)
            exit(0);
        */
        /*
        f0 += (-t0 - s9 + x2 + s12)*s3;
        f1 += (-t1 - s11 - s10 + y2)*s3;
        f2 += (-(-s11 - s10)*s5 - (s9 - s12)*s1)*s3;

        j00 += (1 - 2.F*s0*wInv)*s13;
        j01 -= s5*s1*s2;
        j02 += (-s11 - s10 + 2.F*s8*s5*wInv)*s13;
        j11 += (1 - 2.F*s1*s1*wInv)*s13;
        j12 += (s9 - s12 + 2.F*s8*s1*wInv)*s13;
        j22 += (s4*s4 - s4*s1 + s6*s6 - s6*s5 - 2.F*s8*s8*wInv)*s13;
        //printf("%f   ", j22);
        */
        
        f0 += weight * (-t0 - s9 + x2 + s12)*s3;
        f1 += weight * (-t1 - s11 - s10 + y2)*s3;
        f2 += weight * (-(-s11 - s10)*s5 - (s9 - s12)*s1)*s3;

        j00 += weight * (1 - 2.F*s0*wInv)*s13;
        j01 -= weight * s5*s1*s2;
        j02 += weight * (-s11 - s10 + 2.F*s8*s5*wInv)*s13;
        j11 += weight * (1 - 2.F*s1*s1*wInv)*s13;
        j12 += weight * (s9 - s12 + 2.F*s8*s1*wInv)*s13;
        j22 += weight * (s4*s4 - s4*s1 + s6*s6 - s6*s5 - 2.F*s8*s8*wInv)*s13;
        
    }

    j01 *= 2.F;
    /*
    printf("fs: %f %f %f\n", f0, f1, f2);
    printf("js: %f %f %f %f %f %f\n", j00, j01, j02, j11, j12, j22);
    */
    float detInv = 1.0F / (j00 * (j22*j11 - j12*j12) - j01 * (j22*j01 - j12*j02) + j02 * (j12*j01-j11*j02));

    float i00 = (j22*j11-j12*j12);
    float i01 = -(j22*j01 - j12*j02);
    float i02 = (j12*j01 - j11*j02);
    float i10 = -(j22*j01 - j02 * j12);
    float i11 = j22 *j00 - j02 * j02;
    float i12 = -(j12 * j00 - j01 * j02);
    float i20 = j12*j01 - j02 * j11;
    float i21 = -(j12*j00 - j02*j01);
    float i22 = j11*j00 - j01*j01 ;

    t0 += detInv * (f0 * i00 + f1 * i01 + f2 * i02);
    t1 += detInv * (f0 * i10 + f1 * i11 + f2 * i12);
    r += detInv * (f0 * i20 + f1 * i21 + f2 * i22);
}

Transformation WelschFit(vector<Point2f> corners1, vector<Point2f> corners2, int length){
    
    vector<float> weights(length);
    fill(weights.begin(), weights.end(), 1.0F);
    
    return WelschFitWeighted(corners1, corners2, length, weights);
}


Transformation WelschFitWeighted(vector<Point2f> corners1, vector<Point2f> corners2, int length, vector<float> weights){

        Transformation start = weightedSvd(corners1, corners2, length, weights);
        float startT0 = start.ux2 - start.cos * start.ux1 + start.sin * start.uy1;
        float startT1 = start.uy2 - start.sin * start.ux1 - start.cos * start.uy1;
        float startR = start.rotation;
      
      float t0 = startT0, t1 = startT1, r = startR;

    for(int i=0;i<NUM_STEPS;i++){
        float w = pow(10, log10(START_W) + ((float)i / ((float)NUM_STEPS-1)) * (log10(END_W) - log10(START_W)));
        newtonIteration2Weighted(t0, t1, r, w, corners1, corners2, weights, length);

        if(fabs(t0 - startT0) > NEWTON_STABILITY_LIMIT || fabs(t1 - startT1) > NEWTON_STABILITY_LIMIT || fabs(r - startR) > NEWTON_STABILITY_LIMIT_ROTATION){
            printf("WARNING: newton's method seems to have gone unstable\n");
            t0 = startT0;
            t1 = startT1;
            r = startR;
            break;
        }
    }

    float ux2 = 0, uy2 = 0, denom = 0;
    for(int i=0;i<length;i++){
        ux2 += corners2[i].x * weights[i];
        uy2 += corners2[i].y * weights[i];
        denom += weights[i];
    }
    ux2 /= denom;
    uy2 /= denom;

    float cosr = cos(r);
    float sinr = sin(r);

    float ux1 = cosr * (ux2 - t0) + sinr * (uy2 - t1);
    float uy1 = -sinr * (ux2-t0) + cosr * (uy2 - t1);

    Transformation result = {ux1, uy1, ux2, uy2, r, cosr, sinr};
    return result;
}