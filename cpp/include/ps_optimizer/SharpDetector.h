#ifndef SHARP_DETECTOR_H_
#define SHARP_DETECTOR_H_

#include <opencv2/core/core.hpp>

float modifiedLaplacian(const cv::Mat& src);
float varianceOfLaplacian(const cv::Mat& src);
float tenengrad(const cv::Mat& src, int ksize);
float normalizedGraylevelVariance(const cv::Mat& src);


bool sharpDetector(const cv::Mat& img, float threshold){
    float measure = modifiedLaplacian(img);
    std::cout << "======> the sharpness measure is " << measure << "." << std::endl;
    if( measure < threshold)
        return false;
    return true;
}


// OpenCV port of 'LAPM' algorithm (Nayar89)
float modifiedLaplacian(const cv::Mat& src)
{
    cv::Mat M = (cv::Mat_<float>(3, 1) << -1, 2, -1);
    cv::Mat G = cv::getGaussianKernel(3, -1, CV_32F);

    cv::Mat Lx;
    cv::sepFilter2D(src, Lx, CV_32F, M, G);

    cv::Mat Ly;
    cv::sepFilter2D(src, Ly, CV_32F, G, M);

    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

    float focusMeasure = cv::mean(FM).val[0];
    return focusMeasure;
}

// OpenCV port of 'LAPV' algorithm (Pech2000)
float varianceOfLaplacian(const cv::Mat& src)
{
    cv::Mat lap;
    cv::Laplacian(src, lap, CV_32F);

    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);

    float focusMeasure = sigma.val[0]*sigma.val[0];
    return focusMeasure;
}

// OpenCV port of 'TENG' algorithm (Krotkov86)
float tenengrad(const cv::Mat& src, int ksize)
{
    cv::Mat Gx, Gy;
    cv::Sobel(src, Gx, CV_32F, 1, 0, ksize);
    cv::Sobel(src, Gy, CV_32F, 0, 1, ksize);

    cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

    float focusMeasure = cv::mean(FM).val[0];
    return focusMeasure;
}

// OpenCV port of 'GLVN' algorithm (Santos97)
float normalizedGraylevelVariance(const cv::Mat& src)
{
    cv::Scalar mu, sigma;
    cv::meanStdDev(src, mu, sigma);

    float focusMeasure = (sigma.val[0]*sigma.val[0]) / mu.val[0];
    return focusMeasure;
}

#endif // SHARP_DETECTOR_H_
