#include "utils.h"
#include <stdio.h>
#include "dll.h"

extern "C" __declspec(dllexport) void TrcaTrain(double* darray, double* pTemplate, double* pU)
{
	std::unique_ptr<SSVEP> data = std::make_unique<SSVEP>();
	std::unique_ptr<PreprocessEngine> pe;
	std::unique_ptr<TrcaEngine> te;

	//@zikai 数据排列方式修改
	//data->data_ = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
	//darray, data->train_len_, data->stimulus_, data->electrodes_, data->samples_);
	//Eigen::Tensor<double, 4> tensor(data->train_len_, data->stimulus_, data->electrodes_, data->samples_);
	/*for (int block = 0; block < data->train_len_; block++) {
		for (int stim = 0; stim < data->stimulus_; stim++) {
			for (int elec = 0; elec < data->electrodes_; elec++) {
				for (int sample = 0; sample < data->samples_; sample++) {
					tensor(block, stim, elec, sample) = *darray;
					darray++;
				}
			}
		}
	}*/
	data->data_ = Eigen::TensorMap<Eigen::Tensor<double, 4>>(darray, data->train_len_, data->stimulus_, data->electrodes_, data->samples_);
	std::string path = "./ori.csv";
	tensor4dToCsv(data->data_, path);

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
	path = "./filterBank.csv";
	tensor4dToCsv(data->train_trials_, path);

	data->calculateTemplates();
	te->fit();

	path = "./templ.csv";
	tensor4dToCsv(data->templates_, path);

	path = "./U.csv";
	tensor4dToCsv(te->U_trca_, path);

	TensorTodArray<4>(data->templates_, pTemplate);
	TensorTodArray<4>(te->U_trca_, pU);
}

extern "C" __declspec(dllexport) void TrcaTest(double* darray, double* pTemplate, double* pU, double* pPred)
{
	std::unique_ptr<SSVEP> data = std::make_unique<SSVEP>();
	std::unique_ptr<PreprocessEngine> pe = std::make_unique<PreprocessEngine>(data.get());
	std::unique_ptr<TrcaEngine> te = std::make_unique<TrcaEngine>(data.get());

	//data->data_ = Eigen::TensorMap<Eigen::Tensor<double, 4>>(
		//darray, data->blocks_, data->stimulus_, data->electrodes_, data->samples_);
	//data->templates_ = *p_templates;
	Eigen::Tensor<double, 4> tensor(data->test_len_, data->stimulus_, data->electrodes_, data->samples_);
	for (int block = 0; block < data->test_len_; block++) {
		for (int stim = 0; stim < data->stimulus_; stim++) {
			for (int elec = 0; elec < data->electrodes_; elec++) {
				for (int sample = 0; sample < data->samples_; sample++) {
					tensor(block, stim, elec, sample) = *darray;
					darray++;
				}
			}
		}
	}
	data->data_ = tensor;

	Eigen::Tensor<double, 4> Template(data->stimulus_, data->subbands_, data->electrodes_, data->samples_);
	Eigen::Tensor<double, 4> U(data->subbands_, data->stimulus_, data->electrodes_, 1);
	for (int subband = 0; subband < data->subbands_; subband++) {
		for (int stim = 0; stim < data->stimulus_; stim++) {
			for (int elec = 0; elec < data->electrodes_; elec++) {
				U(subband, stim, elec, 0) = *pU;
				pU++;
				for (int sample = 0; sample < data->samples_; sample++) {
					Template(stim, subband, elec, sample) = *pTemplate;
					pTemplate++;
				}
			}
		}
	}

	data->templates_ = Template;
	te->U_trca_ = U;

	//@zikai test_trials init
	//@zikai get single trial logic 要改
	for (int block = 0; block < data->test_len_; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> single_trial = data->getSingleTrial(block, stimulus);
			single_trial = pe->notch(single_trial);
			data->test_trials_.chip<0>(block * data->stimulus_ + stimulus) = pe->filterBank(single_trial);
		}
	}

	TensorTodArray<1>(te->predict(), pPred);
}
