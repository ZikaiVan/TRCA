#include "PreprocessEngine.h"
#include <cmath>
#include <iostream>
#define M_PI 3.1415926

PreprocessEngine::~PreprocessEngine(){}

PreprocessEngine::PreprocessEngine(SSVEP* data) {
    s_rate = data->s_rate_;
    bsf = nullptr;
}

void PreprocessEngine::notch(SSVEP* data) {
    bsf = new Cheby1BSF(4, 2, 47, 53, s_rate);
    Eigen::Tensor<double, 2> trial = data->getSingleTrial(0, 0, 125, 660); //还没处理数据范围
    filtFilt(trial);
}

void PreprocessEngine::filterBank(SSVEP* data) {

}

void PreprocessEngine::filtFilt(Eigen::Tensor<double, 2> trial) {
    int edge = 3 * (std::max(bsf->b_.size(), bsf->a_.size()) - 1); 
    Eigen::Tensor<double, 2> ext = oddExt(trial, edge, 1);
    Eigen::Tensor<double, 1> zi = lFilterZi();

    //Eigen::Tensor<double, 2> ext = oddExt(data->data, padlen, 1);
    //def filtfilt(b, a, x, axis = -1, padtype = 'odd', padlen = 3*(max(len(b1),len(a1))-1)), method = 'pad',
      //  irlen = None) :
}

Eigen::Tensor<double, 2> PreprocessEngine::oddExt(const Eigen::Tensor<double, 2>& x, int edge, int axis) {
    Eigen::Tensor<double, 2> left_ext = axisSlice(x, 1, edge + 1, -1, axis);
    Eigen::Tensor<double, 2> left_end = axisSlice(x, 0, 1, 1, axis).broadcast(Eigen::array<int, 2>({ 1, int(left_ext.dimension(1))}));
    Eigen::Tensor<double, 2> right_ext = axisSlice(x, x.dimension(1) -(edge + 1), x.dimension(1) - 1, -1, axis);
    Eigen::Tensor<double, 2> right_end = axisSlice(x, x.dimension(1) - 1, x.dimension(1), 1, axis).broadcast(Eigen::array<int, 2>({ 1, int(right_ext.dimension(1))}));
    Eigen::Tensor<double, 2> ext = (2 * left_end - left_ext).concatenate(x, axis).concatenate(2 * right_end - right_ext, axis);
    //std::cout << ext;
    return ext;
}

Eigen::Tensor<double, 2> PreprocessEngine::axisSlice(const Eigen::Tensor<double, 2>& a, int start, int stop, int step, int axis) {
    Eigen::array<Eigen::Index, 2> dims = a.dimensions();
    Eigen::array<Eigen::Index, 2> start_indices = { 0, start };
    Eigen::array<Eigen::Index, 2> stop_indices = { dims[0], stop - start };
    Eigen::array<Eigen::Index, 2> strides = { 1, 1 };

    Eigen::Tensor<double, 2> b = a.slice(start_indices, stop_indices).stride(strides);
    if (step < 0) {
        Eigen::Tensor<double, 2> c = b.reverse(Eigen::array<bool, 2>({false, true}));
        b = c;
    }
    return b;
}

Eigen::Tensor<double, 1> PreprocessEngine::lFilterZi() {
    Eigen::Tensor<double, 1> b_copy = bsf->b_;
    Eigen::Tensor<double, 1> a_copy = bsf->a_;

    while (a_copy.dimension(0) > 1 && a_copy(0) == 0.0) {
        Eigen::array<Eigen::Index, 1> start_indices = { 1 };
        Eigen::array<Eigen::Index, 1> stop_indices = { a_copy.dimension(0) };
        a_copy = a_copy.slice(start_indices, stop_indices);
    }

    if (a_copy.dimension(0) < 1) {
        throw std::invalid_argument("There must be at least one nonzero `a` coefficient.");
    }

    if (a_copy(0) != 1.0) {
        // Normalize the coefficients so a[0] == 1.
        b_copy = b_copy / a_copy(0);
        a_copy = a_copy / a_copy(0);
    }

    int n = std::max(a_copy.size(), b_copy.size());

    // Pad a or b with zeros so they are the same length.
    if (a_copy.size() < n) {
        Eigen::Tensor<double, 1> zeros(n - a_copy.size());
        Eigen::Tensor<double, 1> aa_copy = a_copy.concatenate(zeros.setZero(), 0);
        a_copy = aa_copy;
    }
    else if (b_copy.size() < n) {
        Eigen::Tensor<double, 1> zeros(n - b_copy.size());
        Eigen::Tensor<double, 1> bb_copy = b_copy.concatenate(zeros.setZero(), 0);
        b_copy = bb_copy;
    }

    Eigen::MatrixXd IminusA = Eigen::MatrixXd::Identity(n - 1, n - 1) - linalg.companion(a_copy).transpose().eval();
    Eigen::Tensor<double, 1> B = b_copy.slice(1, b_copy.dimension(0)) - a_copy.slice(1, a_copy.dimension(0)) * b_copy(0);
    // Solve zi = A*zi + B
    Eigen::Tensor<double, 1> zi = IminusA.fullPivLu().solve(B);

    return b_copy;
}
