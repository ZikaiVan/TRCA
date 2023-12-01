#pragma once
#include <StdVector>
#include "Cheby1Filter.h"
#include "SSVEP.h"
#include "utils.h"

class PreprocessEngine {
public:
	~PreprocessEngine();
	PreprocessEngine(const SSVEP* data);
	Eigen::Tensor<double, 2> notch(const Eigen::Tensor<double, 2>& trial);
	Eigen::Tensor<double, 3> filterBank(const Eigen::Tensor<double, 2>& trial);

private:
	const SSVEP* data_;
	int s_rate_;
	Cheby1Filter* bsf_;
	Cheby1Filter* bpf_;

	Eigen::Tensor<double, 2> filtFilt(const Eigen::Tensor<double, 2>& trial, const Cheby1Filter& filter, int axis);
	Eigen::Tensor<double, 2> oddExt(const Eigen::Tensor<double, 2>& x, int edge, int axis = 1);
	Eigen::Tensor<double, 2> axisSlice(const Eigen::Tensor<double, 2>& a, int start = 0, int stop = 1, int step = 1, int axis = 1);
	Eigen::Tensor<double, 2> lFilterZi(const Cheby1Filter& filter);
	Eigen::Tensor<double, 2> lFilter(const Eigen::Tensor<double, 2>& x, Eigen::Tensor<double, 2> z, const Cheby1Filter& filter);
};
