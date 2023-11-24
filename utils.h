#pragma once
#include <Tensor>
#include <fstream>
#include <string>

void tensor2dToCsv(const Eigen::Tensor<double, 2>& tensor, int flag = 0, const std::string& path = "../cpp.csv");

