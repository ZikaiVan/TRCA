#include "utils.h"
#include <stdio.h>
#include "dll.h"

extern "C" __declspec(dllexport) void TrcaTrain(double* darray, double* pTemplate, double* pU)
{
	std::unique_ptr<SSVEP> data = std::make_unique<SSVEP>();
	std::unique_ptr<PreprocessEngine> pe;
	std::unique_ptr<TrcaEngine> te;

//行优先->列优先：转置
//列优先->行优先：reshape，其中double需要cast到float上面才能reshape，
//	reshape之后要赋值给tensor float变量之后才能cast到double，原因是reshape传回参数不能被cast解析
	Eigen::Tensor<double, 4, Eigen::RowMajor> input = Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>>(
		darray, data->train_len_, data->stimulus_, data->electrodes_, data->samples_);
	data->data_ = input.swap_layout().shuffle(Eigen::array<int, 4>{3,2,1,0});

	pe = std::make_unique<PreprocessEngine>(data.get());
	te = std::make_unique<TrcaEngine>(data.get());

	//@zikai 4维：(训练次数，目标数，电极通道数，单通道数据)
	//@zikai train_trials init，优化trials分割
	for (int block = 0; block < data->train_len_; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> single_trial = data->getSingleTrial(block, stimulus);
			single_trial = pe->notch(single_trial);
			data->train_trials_.chip<0>(block * data->stimulus_ + stimulus) = pe->filterBank(single_trial);
		}
	}

	data->calculateTemplates();
	te->fit();

	memcpy(pTemplate, data->templates_.data(), data->templates_.size() * sizeof(double));
	memcpy(pU, te->U_trca_.data(), te->U_trca_.size() * sizeof(double));
}

extern "C" __declspec(dllexport) void TrcaTest(double* darray, double* pTemplate, double* pU, double* pPred)
{
	std::unique_ptr<SSVEP> data = std::make_unique<SSVEP>();
	std::unique_ptr<PreprocessEngine> pe = std::make_unique<PreprocessEngine>(data.get());
	std::unique_ptr<TrcaEngine> te = std::make_unique<TrcaEngine>(data.get());

	Eigen::Tensor<double, 4, Eigen::RowMajor> input = Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>>(
		darray, data->test_len_, data->stimulus_, data->electrodes_, data->samples_);
	data->data_ = input.swap_layout().shuffle(Eigen::array<int, 4>{3, 2, 1, 0});
	data->templates_ = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
		pTemplate, data->subbands_, data->stimulus_, data->electrodes_, data->samples_);
	te->U_trca_ = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
		pU, data->subbands_, data->stimulus_, data->electrodes_, 1);

	//@zikai test_trials init
	//@zikai get single trial logic 要改
	for (int block = 0; block < data->test_len_; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> single_trial = data->getSingleTrial(block, stimulus);
			single_trial = pe->notch(single_trial);
			data->test_trials_.chip<0>(block * data->stimulus_ + stimulus) = pe->filterBank(single_trial);
		}
	}

	memcpy(pPred, te->predict().data(), te->predict().size() * sizeof(double));
}
