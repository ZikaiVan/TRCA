#pragma once
#include <StdVector>
#include "Cheby1BSF.h"
#include "SSVEP.h"

class PreprocessEngine
{
public:
	~PreprocessEngine();
	PreprocessEngine(SSVEP* data);
	void notch(SSVEP* data);
	void filterBank(SSVEP* data);

private:
	double s_rate;
	Cheby1BSF* bsf;
	void filtFilt(Eigen::Tensor<double, 2> trial);
	Eigen::Tensor<double, 2> oddExt(const Eigen::Tensor<double, 2>& x, int edge, int axis = 1);
	Eigen::Tensor<double, 2> axisSlice(const Eigen::Tensor<double, 2>& a, int start = 0, int stop = 1, int step = 1, int axis = 1);
	Eigen::Tensor<double, 1> lFilterZi();
};
