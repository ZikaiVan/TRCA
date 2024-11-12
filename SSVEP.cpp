#include "SSVEP.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include "utils.h"

// public
SSVEP::SSVEP(int train_len, int s_rate, double duration, int subbands, int electrodes, int stimulus) {
    train_len_ = train_len;
    s_rate_ = s_rate;
    duration_ = duration;
    samples_ = duration * s_rate_;
    subbands_ = subbands;
    electrodes_ = electrodes;
    stimulus_ = stimulus;

}

SSVEP::SSVEP(int s_rate, double duration, int subbands, int electrodes, int stimulus) {
    s_rate_ = s_rate;
    duration_ = duration;
    samples_ = duration * s_rate_;
    subbands_ = subbands;
    electrodes_ = electrodes;
    stimulus_ = stimulus;
}

Eigen::Tensor<double, 4> calculateTemplates(Eigen::Tensor<double, 4>& tensor, int stimulus_, int train_len_) {
    Eigen::Tensor<double, 4> templates(stimulus_, tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
    for (int i = 0; i < stimulus_; ++i) {
        int k = 0;
        Eigen::Tensor<double, 4> template_trials(train_len_, tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
        for (int j = i; j < tensor.dimension(0); j += stimulus_, k++) {
            template_trials.chip<0>(k) = tensor.chip<0>(j);
        }
        templates.chip<0>(i) = template_trials.mean(Eigen::array<Eigen::DenseIndex, 1>({ 0 }));
    }
    return templates;
}

Eigen::Tensor<double, 2> tensor1to2(const Eigen::Tensor<double, 1>& tensor) {
    Eigen::Tensor<double, 2> tensor1(1, tensor.dimension(0));
    tensor1.chip<0>(0) = tensor;
    return tensor1;
}

Eigen::Tensor<double, 2> transpose(const Eigen::Tensor<double, 2>& tensor) {
    Eigen::array<int, 2> shuffleOrder({ 1, 0 });
    return tensor.shuffle(shuffleOrder);
}

Eigen::Tensor<double, 2> rowCompanion(const Eigen::Tensor<double, 2>& tensor) {
    int n = tensor.dimension(1);
    Eigen::Tensor<double, 2> companion(n - 1, n - 1);
    companion.setZero();
    for (int i = 0; i < n - 2; i++) {
        companion(i + 1, i) = 1.0;
    }
    for (int i = 0; i < n-1; i++) {
        companion(0, i) = -tensor(0, i+1) / tensor(0, 0);
    }
    return companion;
}

Eigen::Tensor<double, 2> identity(int n) {
    Eigen::Tensor<double, 2> I(n, n);
    I.setZero();
    for (int i = 0; i < n; i++) {
        I(i, i) = 1.0;
    }
    return I;
}

Eigen::Tensor<double, 2> solveZi(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> A_(A.data(), A.dimension(0), A.dimension(1));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> B_(B.data(), B.dimension(0), B.dimension(1));
    Eigen::VectorXd zi = A_.fullPivLu().solve(B_);

    Eigen::Tensor<double, 2> tensor(1, zi.size());
    for (int i = 0; i < zi.size(); i++) {
        tensor(0, i) = zi(i);
    }
    return tensor;
}

Eigen::Tensor<double, 2> solveEig(const Eigen::Tensor<double, 2>& S, const Eigen::Tensor<double, 2>& Q) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S_(S.data(), S.dimension(0), S.dimension(1));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Q_(Q.data(), Q.dimension(0), Q.dimension(1));
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver;
    solver.compute(S_, Q_);
    Eigen::VectorXd realEigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd realEigenvectors = solver.eigenvectors().real();

    // Sort the indices in descending order
    std::vector<int> sort_idx(realEigenvalues.size());
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::sort(sort_idx.begin(), sort_idx.end(), [&realEigenvalues](int i1, int i2) {
        return realEigenvalues[i1] > realEigenvalues[i2]; });

    // Reorder the eigenvectors according to the sorted indices
    Eigen::MatrixXd eig_vec = realEigenvectors;
    for (int i = 0; i < sort_idx.size(); ++i) {
        eig_vec.col(i) = realEigenvectors.col(sort_idx[i]);
    }

    // Compute the square values
    Eigen::MatrixXd square_val = (eig_vec.transpose() * Q_ * eig_vec).diagonal();

    // Compute the norm values
    Eigen::VectorXd norm_v = square_val.cwiseSqrt();

    // Normalize the eigenvectors
    for (int i = 0; i < eig_vec.cols(); ++i) {
        eig_vec.col(i) /= norm_v[i];
    }

    Eigen::TensorMap<Eigen::Tensor<double, 2>> tensor(eig_vec.data(), eig_vec.rows(), eig_vec.cols());

    return tensor;
}

Eigen::Tensor<double, 2> vecCov(Eigen::Tensor<double, 1>& a, Eigen::Tensor<double, 1>& b,
    bool rowvar, const std::string& dtype) {
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(a.data(), a.size());
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(b.data(), b.size());
    Eigen::MatrixXd x_mean = x.array() - x.mean();
    Eigen::MatrixXd y_mean = y.array() - y.mean();
    Eigen::Tensor<double, 2> c(2, 2);
    double normalizer = static_cast<double>(x.size() - 1);

    c(0, 0) = (x_mean.array() * x_mean.array()).sum() / normalizer;
    c(0, 1) = (x_mean.array() * y_mean.array()).sum() / normalizer;
    c(1, 0) = c(0, 1);
    c(1, 1) = (y_mean.array() * y_mean.array()).sum() / normalizer;
    return c;
}


