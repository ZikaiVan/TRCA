#ifndef DLL_H
#define DLL_H
#include "PreprocessEngine.h"
#include "TrcaEngine.h"

extern "C" __declspec(dllexport) void TrcaTrain(double* darray, double* pTemplate, double* pU);
extern "C" __declspec(dllexport) void TrcaFit(
	double* array, Eigen::Tensor<double, 4>*&p_templates,
	Eigen::Tensor<double, 4>*&p_U, Eigen::Tensor<double, 1>*&p_pred);

#endif // DLL_H