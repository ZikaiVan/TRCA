#pragma once
#include <Dense>
#include <Core>
#include <Tensor>
#define M_PI 3.1415926

class Cheby1BSF {
public:
    Cheby1BSF(int order, int ripple, double wn1, double wn2, double srate);

    Eigen::Tensor<double, 1> b_;
    Eigen::Tensor<double, 1> a_;

private:
    void calculateZPK();
    void lp2bsZPK();
    void bilinearZPK();
    Eigen::VectorXcd poly(Eigen::VectorXcd x);
    Eigen::VectorXcd convolve(Eigen::VectorXcd x, Eigen::VectorXcd y, int loc);
    Eigen::Tensor<double, 1> vecXcd2Tensor(Eigen::VectorXcd vector);

    int order_;
    double ripple_;
    double warped_[2];
    double srate_;
    double bw_;
    double wo_;
    Eigen::VectorXcd z_;
    Eigen::VectorXcd p_;
    double k_;
};