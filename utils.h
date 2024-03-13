#ifndef UTILS_H
#define UTILS_H
#pragma once
#include <Tensor>
#include <fstream>
#include <string>

void tensor1dToCsv(const Eigen::Tensor<double, 1>& tensor, std::string& path);
void tensor2dToCsv(const Eigen::Tensor<double, 2>& tensor, std::string& path, int flag = 0);
void tensor3dToCsv(const Eigen::Tensor<double, 3>& tensor, std::string& path);
void tensor4dToCsv(const Eigen::Tensor<double, 4>& tensor, std::string& path);
void WriteToFile(double data);

template<int D>
void TensorTodArray(Eigen::Tensor<double, D> tensor, double* array) {
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        array[i] = tensor.data()[i];
    }
}

#endif // UTILS_H

