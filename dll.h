#ifndef DLL_H
#define DLL_H
#include "Preprocess.h"
#include "Trca.h"

extern "C" __declspec(dllexport) int FilterBank(double* darray, double* dout,
	int s_rate, int subbands, int len, int stimulus, int electrodes, int num_samples, int debug);

extern "C" __declspec(dllexport) int TrcaTrain(double* darray, double* pTemplate, double* pU,
	int s_rate, int subbands, int train_len, int stimulus, int electrodes, int num_samples, int debug);
extern "C" __declspec(dllexport) int TrcaTrainOnly(double* darray, double* pTemplate, double* pU,
	int s_rate, int subbands, int train_len, int stimulus, int electrodes, int num_samples, int debug);

extern "C" __declspec(dllexport) int TrcaTest(double* darray, double* pTemplate, double* pU, double* pcoeff, 
	int* pPred, int s_rate, int subbands, int test_len, int stimulus, int electrodes, int num_samples, int debug);
extern "C" __declspec(dllexport) int TrcaTestOnly(double* darray, double* pTemplate, double* pU, double* pcoeff,
	int* pPred, int s_rate, int subbands, int test_len, int stimulus, int electrodes, int num_samples, int debug);
extern "C" __declspec(dllexport) int TrcaTestCsv(double* darray, char* pTemplate, char* pU, double* pcoeff,
	int* pPred, int s_rate, int subbands, int test_len, int stimulus, int electrodes, int num_samples, int debug);
extern "C" __declspec(dllexport) int TrcaTestOnlyCsv(double* darray, char* pTemplate, char* pU, double* pcoeff,
	int* pPred, int s_rate, int subbands, int test_len, int stimulus, int electrodes, int num_samples, int debug);

#endif // DLL_H