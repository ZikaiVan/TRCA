#ifndef CHEBY1FILTER_H
#define CHEBY1FILTER_H
#include <Dense>
#include <Core>
#include <Tensor>
#define M_PI 3.1415926

class Cheby1Filter {
public:
    Eigen::Tensor<double, 1> b_;
    Eigen::Tensor<double, 1> a_;
    Cheby1Filter();
    Cheby1Filter(int order, int ripple, double wn1, double wn2, double srate, char type);

private:
    int order_;
    double ripple_;
    double warped_[2];
    double srate_;
    double bw_;
    double wo_;
    Eigen::VectorXcd z_;
    Eigen::VectorXcd p_;
    double k_;

    void calculateZPK();
    void lp2bpZpk();
    void lp2bsZPK();
    void bilinearZPK();
    Eigen::VectorXcd poly(const Eigen::VectorXcd& x);
    Eigen::VectorXcd convolve(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int loc);
    Eigen::Tensor<double, 1> vecXcd2Tensor(Eigen::VectorXcd vector);
};
#endif // CHEBY1FILTER_H