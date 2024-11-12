#ifndef UTILS_H
#define UTILS_H
#pragma once
#include <Tensor>
#include <string>

// DATA UTILS
Eigen::Tensor<double, 4> calculateTemplates(Eigen::Tensor<double, 4>& tensor, int stimulus_, int train_len_);
Eigen::Tensor<double, 2> tensor1to2(const Eigen::Tensor<double, 1>& tensor1);
Eigen::Tensor<double, 2> transpose(const Eigen::Tensor<double, 2>& tensor);
Eigen::Tensor<double, 2> rowCompanion(const Eigen::Tensor<double, 2>& input);
Eigen::Tensor<double, 2> identity(int n);
Eigen::Tensor<double, 2> solveZi(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B);
Eigen::Tensor<double, 2> solveEig(const Eigen::Tensor<double, 2>& S, const Eigen::Tensor<double, 2>& Q);
Eigen::Tensor<double, 2> vecCov(Eigen::Tensor<double, 1>& a, Eigen::Tensor<double, 1>& b,
	bool rowvar = true, const std::string& dtype = "real");

// DEBUG UTILS
void tensor1dToCsv(const Eigen::Tensor<double, 1>& tensor, const char* path);
void tensor2dToCsv(const Eigen::Tensor<double, 2>& tensor, const char* path, int flag = 0);
void tensor3dToCsv(const Eigen::Tensor<double, 3>& tensor, const char* path, int flag = 0);
void tensor4dToCsv(const Eigen::Tensor<double, 4>& tensor, const char* path);
void WriteToFile(double data);

template<int D>
void TensorTodArray(Eigen::Tensor<double, D> tensor, double* array) {
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        array[i] = tensor.data()[i];
    }
}

#endif // UTILS_H

