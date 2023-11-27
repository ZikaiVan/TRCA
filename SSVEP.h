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
	int latency_;
	int duration_;
	int pre_stim_;
	int rest_;

	int electrodes_;
	int stimulus_;
	int blocks_;
	int sets_;

	int subbands_;
	int samples_;

	SSVEP(const std::string& path = "./config.ini");
	void loadCsv(const std::string& path);
	void loadMat(const std::string& path);

	Eigen::Tensor<double, 2> getSingleTrial(int block, int stimulus);
	Eigen::Tensor<double, 2> tensor1to2(const Eigen::Tensor<double, 1>& tensor1);
	Eigen::Tensor<double, 2> transpose(const Eigen::Tensor<double, 2>& tensor);
	Eigen::Tensor<double, 2> rowCompanion(const Eigen::Tensor<double, 2>& input);
	Eigen::Tensor<double, 2> identity(int n);
	Eigen::Tensor<double, 2> solve(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B);

private:
	void loadConfig(const std::string& path = "./config.ini");
};
