#pragma once
#include <string>
#include <Dense>
#include <Core>
#include <Tensor>

class SSVEP
{
public:
	Eigen::Tensor<double, 4> data;
	int s_rate_;
	double latency_;
	double duration_;
	double pre_stim_;
	double rest_;
	int samples_;
	int electrodes_;
	int stimulus_;
	int blocks_;
	int sets_;

	SSVEP(const std::string& path = "./config.ini");
	void loadCsv(const std::string& path);
	void loadMat(const std::string& path);
	Eigen::Tensor<double, 2> getSingleTrial(int block, int stimulus, int start, int end);

private:
	void loadConfig(const std::string& path = "./config.ini");
};
