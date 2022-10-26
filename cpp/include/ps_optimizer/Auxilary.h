//============================================================================
// Name        :Auxilary.h
// Author      : Lu Sang
// Date        : 09/2021
// License     : GNU General Public License
// Description : implementation of some helper functions
//============================================================================

#ifndef AUXILARY_H_
#define AUXILARY_H_

#include "mat.h"

#include <Eigen/StdVector>
#include <Eigen/Sparse>
// #include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>

//======================================= helper functions ============================================================================

//! check for NaN values inside Eigen object
template<class S> 
bool checkNan(S vector)
{
    return vector.array().isNaN().sum(); 
}

//! skew-symmetric matrix of a 3D vector
static inline Mat3f skew(Vec3f v) {
	Mat3f vx = Mat3f::Zero();
	vx(0,1) = -v[2];
	vx(0,2) =  v[1];
	vx(1,0) =  v[2];
	vx(1,2) = -v[0];
	vx(2,0) = -v[1];
	vx(2,1) =  v[0];
	return vx;
}

//! interpolation for RGB image
static Vec3f interpolateImage(const float m, const float n, const cv::Mat& img)
{
	int x = std::floor(m);
	int y = std::floor(n);
	cv::Vec3f tmp;
	if ((x+1) < img.rows && (y+1) < img.cols){
		tmp = (y+1.0-n)*(m-x)*img.at<cv::Vec3f>(x+1,y) + (y+1.0-n)*(x+1.0-m)*img.at<cv::Vec3f>(x,y) + (n-y)*(m-x)*img.at<cv::Vec3f>(x+1,y+1) + (n-y)*(x+1.0-m)*img.at<cv::Vec3f>(x,y+1);
	}
	else if ((y+1) < img.cols && x >= img.rows){
		tmp = (y+1.0-n)*img.at<cv::Vec3f>(x,y) + (n-y)*img.at<cv::Vec3f>(x,y+1);
	}
	else if ( y >= img.cols && (x+1) < img.rows){
		tmp = (m-x)*img.at<cv::Vec3f>(x+1,y) + (x+1.0-m)*img.at<cv::Vec3f>(x,y);
	}
	else{
		tmp = img.at<cv::Vec3f>(x,y);
	}

	Vec3f intensity(tmp[2],tmp[1],tmp[0]); // OpenCV stores image colors as BGR.
	return intensity;
}

//! gradient of image
static Vec3f computeImageGradient(const float m, const float n, const cv::Mat& img, int direction)
{
	float w00, w01, w10, w11;

	int x = std::floor(m);
	int y = std::floor(n);

	w01 = m - x;
	w11 = n - y;
	w00 = 1.0 - w01;
	w10 = 1.0 - w11;

    // compute gradient manually using finite differences
    
    cv::Vec3f v0, v1;

    if (direction == 0)
    {
		// x-direction
		if( (x+1)<img.rows && (y+1) < img.cols){
		
        	v0 = img.at<cv::Vec3f>(x,y+1) - img.at<cv::Vec3f>(x,y);
        	v1 = img.at<cv::Vec3f>(x+1,y+1) - img.at<cv::Vec3f>(x+1,y);
      
        	return w00 * Vec3f(v0[2],v0[1],v0[0]) + w01 *  Vec3f(v1[2],v1[1],v1[0]);
		}
		else if ( (x+1) >= img.rows){// && (y+1) < img.cols){

			v0 = img.at<cv::Vec3f>(x,y+1) - img.at<cv::Vec3f>(x,y);

			return Vec3f(v0[2],v0[1],v0[0]);
		}
		else if ( (x+1) < img.rows && (y+1) >= img.cols){

			v0 = -img.at<cv::Vec3f>(x,y-1) + img.at<cv::Vec3f>(x,y);
			v1 = -img.at<cv::Vec3f>(x+1,y-1) + img.at<cv::Vec3f>(x+1,y);

			return w00 * Vec3f(v0[2],v0[1],v0[0]) + w01 *  Vec3f(v1[2],v1[1],v1[0]);
		}
    }
    else
    {
        // y-direction
		if ((x+1)<img.rows && (y+1) < img.cols){
        	v0 = img.at<cv::Vec3f>(x+1,y) - img.at<cv::Vec3f>(x,y);
        	v1 = img.at<cv::Vec3f>(x+1,y+1) - img.at<cv::Vec3f>(x,y+1);
        	return w10 * Vec3f(v0[2],v0[1],v0[0]) + w11 * Vec3f(v1[2],v1[1],v1[0]);
		}

		else if ( (x+1) >= img.rows && (y+1) < img.cols){
			v0 = -img.at<cv::Vec3f>(x-1,y) + img.at<cv::Vec3f>(x,y);
        	v1 = -img.at<cv::Vec3f>(x-1,y+1) + img.at<cv::Vec3f>(x,y+1);
			return w10 * Vec3f(v0[2],v0[1],v0[0]) + w11 * Vec3f(v1[2],v1[1],v1[0]);
		}
		else if ( (y+1) >= img.cols){ // && (x+1) < img.rows){
			v0 = img.at<cv::Vec3f>(x+1,y) - img.at<cv::Vec3f>(x,y);
			return Vec3f(v0[2],v0[1],v0[0]);
		}
    }
}

#endif //AUXILARY_H_