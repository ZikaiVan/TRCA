#include "PreprocessEngine.h"
#include <iostream>

int main() {
	int subject = 28;
	SSVEP* data = new SSVEP();
	data->loadCsv("./data/S028.csv");
	PreprocessEngine* pe;
	pe = new PreprocessEngine(data);

	int block = 0;
	int stimulus = 0;
	Eigen::Tensor<double, 2> trial = data->getSingleTrial(block, stimulus, 125, 660);
	trial = pe->notch(trial);
	Eigen::Tensor<double, 3> trial3D(5, 8, 535);
	trial3D.setZero();
	trial3D = pe->filterBank(trial);
	for (int i=0;i<5;i++)
		tensor2dToCsv(trial3D.chip<0>(i), 1);


	return 0;
}

