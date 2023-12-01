#pragma once
#include "SSVEP.h"
#include <Tensor>

class TrcaEngine {
public:
	TrcaEngine();
	~TrcaEngine();
	TrcaEngine(const SSVEP* data);
	void fit();
	void fit(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<int, 1>& labels, const Eigen::Tensor<double, 4>& templates);
	Eigen::Tensor<double, 1> predict() const;
	Eigen::Tensor<double, 1> predict(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates,
		const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const;


private:
	Eigen::Tensor<double, 4> U_trca_;
	Eigen::Tensor<double, 1> filter_banks_weights_;
	Eigen::Tensor<int, 1> possible_classes_;
	const SSVEP* data_;

	Eigen::Tensor<double, 2> trcaU(const Eigen::Tensor<double, 3>& trials) const;
	Eigen::Tensor<double, 2> canoncorrWithUV(const Eigen::Tensor<double, 3>& trials, const Eigen::Tensor<double, 4>& templates,
		const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const;
	Eigen::Tensor<double, 2> corrCoef(Eigen::Tensor<double, 1>& x, Eigen::Tensor<double, 1>& y,
		bool rowvar=true, const std::string& dtype="real") const;
};