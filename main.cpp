#include "PreprocessEngine.h"
#include "TrcaEngine.h"
#include <iostream>
#include <chrono>
#include <windows.h>
#include <psapi.h>

int main() {
	int subject = 28;
	SSVEP* data = new SSVEP();
	PreprocessEngine* pe;
	TrcaEngine* te;
	data->loadCsv("./data/S028.csv");
	pe = new PreprocessEngine(data);
	te = new TrcaEngine(data);

	auto start = std::chrono::high_resolution_clock::now();
	for (int block = 0; block < data->train_len_; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> single_trial = data->getSingleTrial(block, stimulus);
			single_trial = pe->notch(single_trial);
			data->train_trials_.chip<0>(block * data->stimulus_ + stimulus) = pe->filterBank(single_trial);
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Train preprocess elapsed time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

	start = std::chrono::high_resolution_clock::now();
	data->calculateTemplates();
	te->fit();
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Fit elapsed time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

	start = std::chrono::high_resolution_clock::now();
	for (int block = data->train_len_; block < data->blocks_; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> single_trial = data->getSingleTrial(block, stimulus);
			single_trial = pe->notch(single_trial);
			data->test_trials_.chip<0>((block-data->train_len_) * data->stimulus_ + stimulus) = pe->filterBank(single_trial);
		}
	}
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Test preprocess elapsed time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

	start = std::chrono::high_resolution_clock::now();
	Eigen::Tensor<double, 1> pre_labels = te->predict();
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Predict elapsed time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";

	double acc = 0, itr = 0;
	for (int i = 0; i < pre_labels.dimension(0); i++) {
		if (pre_labels(i) == data->test_labels_(i)) {
			acc = acc + double(1) / pre_labels.dimension(0);
		}
	}
	std::cout << "acc: " << acc*100 << "%\n";

	return 0;
}


