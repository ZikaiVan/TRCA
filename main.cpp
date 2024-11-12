#include "Preprocess.h"
#include "Trca.h"
#include <iostream>
#include <chrono>
#include <string>

/*void trcaTest(int subject, std::string path);*/

int main() {
	//@zikai 23.12.2 内存释放问题
	/*getchar();
	for (int subject = 20; subject < 30; subject++) {
		std::cout << "S0" << subject << std::endl;
		std::string path="./data/S0"+std::to_string(subject)+".csv";
		trcaTest(subject, path);
	}*/
	return 0;
}
/*
void trcaTest(int subject, std::string path) {
	SSVEP* data = new SSVEP();
	PreprocessEngine* pe;
	TrcaEngine* te;
	data->loadCsv(path);
	pe = new PreprocessEngine(data);
	te = new TrcaEngine(data);

	//@zikai 4维：(训练次数，目标数，电极通道数，单通道数据)
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

	//return 0;
}*/


