#pragma once
#include <string>
#include <Dense>
#include <Core>
#include <Tensor>

class SSVEP {
public:
	Eigen::Tensor<double, 4> data_;
	Eigen::Tensor<double, 4> train_trials_;
	Eigen::Tensor<int, 1> train_labels_;
	Eigen::Tensor<double, 4> test_trials_;
	Eigen::Tensor<int, 1> test_labels_;
	Eigen::Tensor<double, 4> templates_;
	int s_rate_;

	int train_len_;
	int test_len_;

	int latency_;
	int duration_;
	int pre_stim_;
	int rest_;
	int subbands_;
	int type_;

	int electrodes_;
	int stimulus_;
	int blocks_;
	int sets_;

	int samples_;
	

	SSVEP(const std::string& path = "./config.ini");
	void loadCsv(const std::string& path);
	void loadMat(const std::string& path);

	void calculateTemplates();
	Eigen::Tensor<double, 2> getSingleTrial(int block, int stimulus) const;
	Eigen::Tensor<double, 2> tensor1to2(const Eigen::Tensor<double, 1>& tensor1) const;
	Eigen::Tensor<double, 2> transpose(const Eigen::Tensor<double, 2>& tensor) const;
	Eigen::Tensor<double, 2> rowCompanion(const Eigen::Tensor<double, 2>& input) const;
	Eigen::Tensor<double, 2> identity(int n) const;
	Eigen::Tensor<double, 2> solveZi(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B) const;
	Eigen::Tensor<double, 2> solveEig(const Eigen::Tensor<double, 2>& S, const Eigen::Tensor<double, 2>& Q) const;
	Eigen::Tensor<double, 2> vecCov(Eigen::Tensor<double, 1>& a, Eigen::Tensor<double, 1>& b,
		bool rowvar=true, const std::string& dtype="real") const;

private:
	void loadConfig(const std::string& path = "./config.ini");
};
