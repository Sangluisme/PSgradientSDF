// ============================================================================
// Name        : NormalEstimator.h
// Author      : Christiane Sommer
// Date        : 11/2018
// License     : GNU General Public License
// Description : NormalEstimation class
// ============================================================================

#ifndef CV_NORMAL_ESTIMATOR_H_
#define CV_NORMAL_ESTIMATOR_H_

// libraries
#include <Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cv {

/*
 * NormalEstimator class using the FALS method from Badino et al.
 */
template<typename T>
class NormalEstimator {

    using Mat3T = Eigen::Matrix<T, 3, 3>;

    // problem parameters
    int width_, height_;
    Mat3T K_;
    Size window_size_;
    // values needed for normal computation
    Mat_<T> x0_, y0_;
    Mat_<T> x0_n_sq_inv_, y0_n_sq_inv_, n_sq_inv_;
    Mat_<T> Q11_, Q12_, Q13_, Q22_, Q23_, Q33_;

    // similar to Matlab's meshgrid function
    template <typename U>
    void pixelgrid(U shift_x, U shift_y, Mat_ <U> &u, Mat_ <U> &v) {

        std::vector<U> row, col;
        for (int k = 0; k < width_; ++k)
            row.push_back(k - shift_x);
        for (int k = 0; k < height_; ++k)
            col.push_back(k - shift_y);
        Mat Row(row), Col(col);
        repeat(Row.reshape(1, 1), Col.total(), 1, u);
        repeat(Col.reshape(1, 1).t(), 1, Row.total(), v);

    }

    // compute values needed for normal estimation only once
    void cache() {

        Eigen::Matrix<double, 3, 3> K;
        K = K_.template cast<double>();

        const double fx_inv = 1. / K(0, 0);
        const double fy_inv = 1. / K(1, 1);
        const double cx = K(0, 2);
        const double cy = K(1, 2);

        Mat_<double> x0, y0, x0_sq, y0_sq, x0_y0, n_sq;
        Mat_<double> x0_n_sq_inv, y0_n_sq_inv, n_sq_inv;
        Mat_<double> M11, M12, M13, M22, M23, M33, det, det_inv;
        Mat_<double> Q11, Q12, Q13, Q22, Q23, Q33;

        pixelgrid<double>(cx, cy, x0, y0);

        x0 = fx_inv * x0;
        x0_sq = x0.mul(x0);
        y0 = fy_inv * y0;
        y0_sq = y0.mul(y0);
        x0_y0 = x0.mul(y0);

        n_sq = 1. + x0_sq + y0_sq;
        divide(1., n_sq, n_sq_inv);
        x0_n_sq_inv = x0.mul(n_sq_inv);
        y0_n_sq_inv = y0.mul(n_sq_inv);

        boxFilter(x0_sq.mul(n_sq_inv), M11, -1, window_size_, Point(-1, -1), false);
        boxFilter(x0_y0.mul(n_sq_inv), M12, -1, window_size_, Point(-1, -1), false);
        boxFilter(x0_n_sq_inv, M13, -1, window_size_, Point(-1, -1), false);
        boxFilter(y0_sq.mul(n_sq_inv), M22, -1, window_size_, Point(-1, -1), false);
        boxFilter(y0_n_sq_inv, M23, -1, window_size_, Point(-1, -1), false);
        boxFilter(n_sq_inv, M33, -1, window_size_, Point(-1, -1), false);

        det = M11.mul(M22.mul(M33)) + 2 * M12.mul(M23.mul(M13)) -
              (M13.mul(M13.mul(M22)) + M12.mul(M12.mul(M33)) + M23.mul(M23.mul(M11)));
        divide(1., det, det_inv);

        Q11 = det_inv.mul(M22.mul(M33) - M23.mul(M23));
        Q12 = det_inv.mul(M13.mul(M23) - M12.mul(M33));
        Q13 = det_inv.mul(M12.mul(M23) - M13.mul(M22));
        Q22 = det_inv.mul(M11.mul(M33) - M13.mul(M13));
        Q23 = det_inv.mul(M12.mul(M13) - M11.mul(M23));
        Q33 = det_inv.mul(M11.mul(M22) - M12.mul(M12));
        
        // TODO: write in more concise way!!
        if (std::is_same<T, float>::value) {
            x0.convertTo(x0_, CV_32F);
            y0.convertTo(y0_, CV_32F);
            x0_n_sq_inv.convertTo(x0_n_sq_inv_, CV_32F);
            y0_n_sq_inv.convertTo(y0_n_sq_inv_, CV_32F);
            n_sq_inv.convertTo(n_sq_inv_, CV_32F);
            Q11.convertTo(Q11_, CV_32F);
            Q12.convertTo(Q12_, CV_32F);
            Q13.convertTo(Q13_, CV_32F);
            Q22.convertTo(Q22_, CV_32F);
            Q23.convertTo(Q23_, CV_32F);
            Q33.convertTo(Q33_, CV_32F);
        }
        else {
            x0_ = x0;
            y0_ = y0;
            x0_n_sq_inv_ = x0_n_sq_inv;
            y0_n_sq_inv_ = y0_n_sq_inv;
            n_sq_inv_ = n_sq_inv;
            Q11_ = Q11;
            Q12_ = Q12;
            Q13_ = Q13;
            Q22_ = Q22;
            Q23_ = Q23;
            Q33_ = Q33;
        }
    }

public:

    NormalEstimator(int width, int height, Eigen::Matrix<float, 3, 3> K, Size window_size) :
        width_(width),
        height_(height),
        K_(K.cast<T>()),
        window_size_(window_size)
    {
        cache();
    }

    NormalEstimator(int width, int height, Eigen::Matrix<double, 3, 3> K, Size window_size) :
        width_(width),
        height_(height),
        K_(K.cast<T>()),
        window_size_(window_size)
    {
        cache();
    }

    ~NormalEstimator() {}

    // compute normals
    void compute(const Mat &depth, Mat &nx, Mat &ny, Mat &nz) const {

        // workaround to only divide by depth where it is non-zero
        // not needed for OpenCV versions <4
        Mat_<T> tmp;
        divide(1., depth, tmp);
        Mat z_inv = Mat::zeros(tmp.size(), tmp.type());
        Mat mask = (depth != 0);
        std::cout << mask.type() << std::endl;
        tmp.copyTo(z_inv, mask);

        Mat_<T> b1, b2, b3, norm_n;

        boxFilter(x0_n_sq_inv_.mul(z_inv), b1, -1, window_size_, Point(-1, -1), false);
        boxFilter(y0_n_sq_inv_.mul(z_inv), b2, -1, window_size_, Point(-1, -1), false);
        boxFilter(n_sq_inv_.mul(z_inv), b3, -1, window_size_, Point(-1, -1), false);

        nx = b1.mul(Q11_) + b2.mul(Q12_) + b3.mul(Q13_);
        ny = b1.mul(Q12_) + b2.mul(Q22_) + b3.mul(Q23_);
        nz = b1.mul(Q13_) + b2.mul(Q23_) + b3.mul(Q33_);

        sqrt(nx.mul(nx) + ny.mul(ny) + nz.mul(nz), norm_n);

        divide(nx, norm_n, nx);
        divide(ny, norm_n, ny);
        divide(nz, norm_n, nz);
    }

    Mat *x0_ptr() { return &x0_; }

    Mat *y0_ptr() { return &y0_; }
    
    Mat *n_sq_inv_ptr() { return &n_sq_inv_; }

};

} // namespace cv

#endif // CV_NORMAL_ESTIMATOR_H_
