#include <iostream>
#include <cmath>
#include "Cheby1Filter.h"
#define M_PI 3.1415926

Cheby1Filter::Cheby1Filter(){}

// type: Lowpass, Highpass, bandPass, bandStop
Cheby1Filter::Cheby1Filter(int order, double ripple, double wn1, double wn2, double srate, char type) {
    order_ = order; 
    ripple_ = ripple;
    srate_ = srate; 
    warped_[0] = 2 * 2 * tan(M_PI * wn1 / srate_);
    warped_[1] = 2 * 2 * tan(M_PI * wn2 / srate_);
    bw_ = warped_[1] - warped_[0];
    wo_ = sqrt(warped_[0] * warped_[1]);

    calculateZPK();
    if (type == 's') {
        lp2bsZPK();
    }
    else if (type == 'p') {
        lp2bpZpk(); 
    }
    
    bilinearZPK();
    b_ = vecXcd2Tensor(k_ * poly(z_));
    a_ = vecXcd2Tensor(poly(p_));

    if (a_(0) != 1.0) {
        // Normalize the coefficients so a[0] == 1.
        b_ = b_ / a_(0);
        a_ = a_ / a_(0);
    }

    // Pad a or b with zeros so they are the same length.
    int n = std::max(a_.size(), b_.size());
    if (a_.size() < n) {
        Eigen::Tensor<double, 1> zeros(n - a_.size());
        Eigen::Tensor<double, 1> a_copy = a_.concatenate(zeros.setZero(), 0);
        a_ = a_copy;
    }
    else if (b_.size() < n) {
        Eigen::Tensor<double, 1> zeros(n - b_.size());
        Eigen::Tensor<double, 1> b_copy = b_.concatenate(zeros.setZero(), 0);
        b_ = b_copy;
    }
    //std::cout << "a:\n" << a_ << "\n\nb:\n" << b_;
}

Eigen::Tensor<double, 1> Cheby1Filter::vecXcd2Tensor(Eigen::VectorXcd vector) {
    Eigen::Tensor<double, 1> tensor(vector.size());
    for (int i = 0; i < vector.size(); i++){
        tensor(i) = vector[i].real();
    }
    return tensor;
}

void Cheby1Filter::calculateZPK() {
    int N = order_;
    double rp = ripple_;
    if (abs(int(N)) != N) {
        throw std::invalid_argument("Filter order must be a nonnegative integer");
    }
    else if (N == 0) {
        k_ = 10 * pow(10, -rp / 20);
        return;
    }

    // Arrange poles in an ellipse on the left half of the S-plane
    Eigen::VectorXd m = Eigen::VectorXd::LinSpaced(N, -N + 1, N - 1);
    Eigen::VectorXd theta = M_PI * m / (2 * N);

    // Ripple factor (epsilon)
    double eps = sqrt(pow(10, 0.1 * rp) - 1.0);
    double mu = 1.0 / N * asinh(1 / eps);
    Eigen::VectorXd mu_vector = Eigen::VectorXd::Ones(m.rows()) * mu;
    p_ = -1 * (mu_vector.cast<std::complex<double>>()
        + std::complex<double>(0, 1) * theta.cast<std::complex<double>>()).array().sinh();

    if (N % 2 == 0) {
        k_ = p_.prod().real() / sqrt((1 + eps * eps));
    }
    else {
        k_ = p_.prod().real();
    }

    z_ = Eigen::VectorXd::Zero(0);
}

void Cheby1Filter::lp2bpZpk() {
    int degree = p_.size() - z_.size();
    Eigen::VectorXcd z_lp = bw_ / 2.0 * z_.array();
    Eigen::VectorXcd p_lp = bw_ / 2.0 * p_.array();
    Eigen::VectorXcd z_bp = Eigen::VectorXcd::Zero(2 * z_lp.size() + degree);
    Eigen::VectorXcd p_bp = Eigen::VectorXcd::Zero(2 * p_lp.size());
    for (int i = 0; i < z_.size(); i++) {
        std::complex<double> sqrt_term = std::sqrt(std::pow(z_lp(i), 2) - std::pow(wo_, 2));
        z_bp(i) = z_lp(i) + sqrt_term;
        z_bp(z_.size() + i) = z_lp(i) - sqrt_term;
    }
    for (int i = 0; i < p_.size(); i++) {
        std::complex<double> sqrt_term = std::sqrt(std::pow(p_lp(i), 2) - std::pow(wo_, 2));
        p_bp(i) = p_lp(i) + sqrt_term;
        p_bp(p_.size() + i) = p_lp(i) - sqrt_term;
    }
    for (int i = 0; i < degree; i++) {
        z_bp(z_bp.size() - degree + i) = 0;
    }
    k_ = k_ * std::pow(bw_, degree);
    z_ = z_bp;
    p_ = p_bp;
}

void Cheby1Filter::lp2bsZPK() {
    int degree = p_.size() - z_.size();
    Eigen::VectorXcd z_hp = bw_ / 2.0 / z_.array();
    Eigen::VectorXcd p_hp = bw_ / 2.0 / p_.array();
    Eigen::VectorXcd z_bs = Eigen::VectorXcd::Zero(2 * z_hp.size() + 2 * degree);
    Eigen::VectorXcd p_bs = Eigen::VectorXcd::Zero(2 * p_hp.size());
    for (int i = 0; i < z_.size(); i++) {
        std::complex<double> sqrt_term = std::sqrt(std::pow(z_hp(i), 2) - std::pow(wo_, 2));
        z_bs(i) = z_hp(i) + sqrt_term;
        z_bs(z_.size() + i) = z_hp(i) - sqrt_term;
    }
    for (int i = 0; i < p_.size(); i++) {
        std::complex<double> sqrt_term = std::sqrt(std::pow(p_hp(i), 2) - std::pow(wo_, 2));
        p_bs(i) = p_hp(i) + sqrt_term;
        p_bs(p_.size() + i) = p_hp(i) - sqrt_term;
    }
    for (int i = 0; i < degree; i++) {
        z_bs(z_bs.size() - 2 * degree + i) = std::complex<double>(0, wo_);
    }
    for (int i = 0; i < degree; i++) {
        z_bs(z_bs.size() - degree + i) = std::complex<double>(0, -wo_);
    }
    k_ = k_ * std::real((-1.0 * z_.prod()) / (-1.0 * p_.prod()));
    z_ = z_bs;
    p_ = p_bs;
}

void Cheby1Filter::bilinearZPK() {
    int degree = p_.size() - z_.size();

    double fs2 = 4.0;

    // Bilinear transform the poles and zeros
    Eigen::VectorXcd z_z = (fs2 + z_.array()) / (fs2 - z_.array());
    Eigen::VectorXcd p_z = (fs2 + p_.array()) / (fs2 - p_.array());

    // Any zeros that were at infinity get moved to the Nyquist frequency
    Eigen::VectorXd ones_vec = Eigen::VectorXd::Ones(degree);
    z_z.conservativeResize(z_z.size() + degree);
    z_z.tail(degree) = -ones_vec;

    // Compensate for gain change
    k_ = k_ * (fs2 - z_.array()).prod().real() / (fs2 - p_.array()).prod().real();
    z_ = z_z;
    p_ = p_z;
}

Eigen::VectorXcd Cheby1Filter::poly(const Eigen::VectorXcd& x) {
    Eigen::VectorXcd a = Eigen::VectorXcd::Ones(x.size() + 1);
    Eigen::VectorXcd b = Eigen::VectorXcd::Ones(2);
    for (int i = 0; i < x.size(); i++) {
        b(1) = -x(i);
        /* 23.11.13 @zikai: 循环卷积虚部计算有问题，但实部没问题
        real part convolution correct while wrong in imagery part.*/
        a = convolve(a, b, i + 1);
    }
    return a.real();
}

Eigen::VectorXcd Cheby1Filter::convolve(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int loc) {
    int n = loc + y.size() - 1;
    Eigen::VectorXcd out(n);
    for (int i = 0; i < n; i++) {
        out(i) = 0;
        for (int j = 0; j < loc; j++) {
            if (i - j >= 0 && i - j < y.size()) {
                out(i) += x(j) * y(i - j);
            }
        }
    }
    return out;
}
