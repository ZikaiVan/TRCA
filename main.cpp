#include "PreprocessEngine.h"
#include <iostream>
#include <time.h>
clock_t start;

int main() {
	int subject = 28;
	int train_blocks[7] = { 0,1,2,3,4,5,6 };
	int test_blocks[3] = { 7,8,9 };
	int train_len = sizeof(train_blocks) / sizeof(train_blocks[0]);
	int test_len = sizeof(test_blocks) / sizeof(test_blocks[0]);
	SSVEP* data = new SSVEP();
	PreprocessEngine* pe;
	data->loadCsv("./data/S028.csv");
	pe = new PreprocessEngine(data);

	Eigen::Tensor<double, 4> train4d(train_len * data->stimulus_, data->subbands_, data->electrodes_, data->duration_);
	start = clock();
	for (int block = 0; block < train_len; block++) {
		for (int stimulus = 0; stimulus < data->stimulus_; stimulus++) {
			Eigen::Tensor<double, 2> trial = data->getSingleTrial(train_blocks[block], stimulus);
			// @zikai 11.27 可以改成一次解决所有维度
			trial = pe->notch(trial);
			train4d.chip<0>(block * data->stimulus_ + stimulus) = pe->filterBank(trial);
		}
	}
	std::cout << (clock() - start) / CLOCKS_PER_SEC;


	return 0;
}

