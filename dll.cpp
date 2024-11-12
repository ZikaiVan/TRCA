#include "utils.h"
#include <stdio.h>
#include "dll.h"

extern "C" __declspec(dllexport) int TrcaTrain(double* darray, double* pTemplate, double* pU,
	int s_rate, int subbands, int train_len, int stimulus, int electrodes, int num_samples)
{
	std::unique_ptr<Preprocess> pe = std::make_unique<Preprocess>(s_rate, subbands, electrodes, num_samples);
	std::unique_ptr<Trca> te = std::make_unique<Trca>(subbands, stimulus, electrodes, num_samples, train_len);
	Eigen::Tensor<double, 4> train_trials = Eigen::Tensor<double, 4>(train_len * stimulus, subbands, electrodes, num_samples);
	Eigen::Tensor<double, 4> templates = Eigen::Tensor<double, 4>(stimulus, subbands, electrodes, num_samples);
	Eigen::Tensor<double, 4> U_trca_ = Eigen::Tensor<double, 4>(subbands, stimulus, electrodes, 1);

//行优先->列优先：转置
//列优先->行优先：reshape，其中double需要cast到float上面才能reshape，
//	reshape之后要赋值给tensor float变量之后才能cast到double，原因是reshape传回参数不能被cast解析
	Eigen::Tensor<double, 4, Eigen::RowMajor> dtensor = Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>>(
		darray, train_len, stimulus, electrodes, num_samples);
	Eigen::Tensor<double, 4> input = dtensor.swap_layout().shuffle(Eigen::array<int, 4>{3,2,1,0});

	//@zikai 4维：(训练次数，目标数，电极通道数，单通道数据)
	//@zikai train_trials init，优化trials分割
	for (int block = 0; block < train_len; block++) {
		for (int i = 0; i < stimulus; i++) {
			Eigen::Tensor<double, 2> single_trial = pe->notch(input.chip(block, 0).chip(i, 0));
			train_trials.chip<0>(block * stimulus + i) = pe->filterBank(single_trial);
		}
	}
	templates = calculateTemplates(train_trials, stimulus, train_len);
	U_trca_ = te->fit(train_trials, templates);

	memcpy(pTemplate, templates.data(), templates.size() * sizeof(double));
	memcpy(pU, U_trca_.data(), U_trca_.size() * sizeof(double));
	return 0;
}

extern "C" __declspec(dllexport) int TrcaTest(double* darray, double* pTemplate, double* pU, int* pPred,
	int s_rate, int subbands, int test_len, int stimulus, int electrodes, int num_samples)
{	
	std::unique_ptr<Preprocess> pe = std::make_unique<Preprocess>(s_rate, subbands, electrodes, num_samples);
	std::unique_ptr<Trca> te = std::make_unique<Trca>(subbands, stimulus, electrodes, num_samples);
	Eigen::Tensor<double, 4> test_trial = Eigen::Tensor<double, 4>(test_len, subbands, electrodes, num_samples);

	Eigen::Tensor<double, 3, Eigen::RowMajor> dtensor = Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(
		darray, test_len, electrodes, num_samples);
	Eigen::Tensor<double, 3> input = dtensor.swap_layout().shuffle(Eigen::array<int, 3>{2, 1, 0});
	Eigen::Tensor<double, 4> templates = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
		pTemplate, subbands, stimulus, electrodes, num_samples);
	Eigen::Tensor<double, 4> U_trca = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
		pU, subbands, stimulus, electrodes, 1);

	//@zikai test_trials init
	for (int i = 0; i < test_len; i++) {
		Eigen::Tensor<double, 2> single_trial = pe->notch(input.chip(i, 0));
		test_trial.chip<0>(i) = pe->filterBank(single_trial);
	}

	Eigen::Tensor<int, 1> pred = te->predict(test_trial, templates, U_trca, U_trca);
	memcpy(pPred, pred.data(), pred.size() * sizeof(int));
	return 0;
}
