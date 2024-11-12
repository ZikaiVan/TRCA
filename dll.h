#ifndef DLL_H
#define DLL_H
#include "Preprocess.h"
#include "Trca.h"

extern "C" __declspec(dllexport) int TrcaTrain(double* darray, double* pTemplate, double* pU,
	int s_rate, int subbands, int train_len, int stimulus, int electrodes, int num_samples);
extern "C" __declspec(dllexport) int TrcaTest(double* darray, double* pTemplate, double* pU, int* pPred,
	int s_rate, int subbands, int stimulus, int electrodes, int num_samples);

#endif // DLL_H