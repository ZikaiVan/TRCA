#include "TrcaEngine.h"
#include "utils.h"
#include <iostream>

TrcaEngine::TrcaEngine() {}

TrcaEngine::~TrcaEngine() {}

TrcaEngine::TrcaEngine(const SSVEP* data) {
	data_ = data;
	Eigen::Tensor<double, 4> U_trca(data_->subbands_, data_->stimulus_, data_->electrodes_, 1);
	Eigen::Tensor<double, 1> filter_banks_weights(data_->subbands_);
	Eigen::Tensor<int, 1> possible_classes(data_->stimulus_);
	U_trca.setZero();
	U_trca_ = U_trca;
	for (int i = 0; i < data_->stimulus_; i++) {
		possible_classes(i) = i;
	}
	possible_classes_ = possible_classes;
	if (data_->type_ == 1) {
		for (int i = 0; i < data_->subbands_; i++) {
			filter_banks_weights(i) = pow(i+1, -1.75) + 0.5;
		}
	}
	else {
		for (int i = 0; i < data_->subbands_; i++) {
			filter_banks_weights(i) = pow(i+1, -1.25) + 0.15;
		}
	}
	filter_banks_weights_ = filter_banks_weights;
}

void TrcaEngine::fit(){
	fit(data_->train_trials_, data_->train_labels_, data_->templates_);
}


void TrcaEngine::fit(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<int, 1>& labels, const Eigen::Tensor<double, 4>& templates) {
	// @zikai 23.11.29 Here we fixed component = 1.
	int component = 1;
	Eigen::Tensor<double, 4> U_trca(data_->subbands_, data_->stimulus_, data_->electrodes_, component);
	for (int i = 0; i < data_->subbands_; ++i) {
		Eigen::Tensor<double, 4> trains(data_->stimulus_, data_->train_len_, data_->electrodes_, data_->duration_);
		for (int j = 0; j < data_->stimulus_; j++) {
			int m = 0; // calculate train blocks
			for (int k = j; k < trials.dimension(0); k += data_->stimulus_, m++) {
				trains.chip<0>(j).chip<0>(m) = trials.chip<0>(k).chip<0>(i);
			}
		}
		Eigen::Tensor<double, 3> U(trains.dimension(0), data_->electrodes_, data_->electrodes_);
		for (int j = 0; j < trains.dimension(0); j++) {
			// @zikai 23.11.29: different from MATLAB in positions and multiples.
			U.chip<0>(j) = trcaU(trains.chip<0>(j));
		}
		for (int j = 0; j < data_->stimulus_; j++) {
			U_trca.chip<0>(i).chip<0>(j) = U.chip<0>(j)
				.slice(Eigen::array<int, 2>({ 0, 0 }), Eigen::array<int, 2>({ data_->electrodes_, component }));
		}
		// @zikai due to the differece, maybe some wrong result will generate, hasn't checked.
		U_trca_ = U_trca;
	}
}

Eigen::Tensor<double, 2> TrcaEngine::trcaU(const Eigen::Tensor<double, 3>& trials) const {
	Eigen::Tensor<double, 2> trca_X1(trials.dimension(1), trials.dimension(2));
	Eigen::Tensor<double, 3> trca_X2_tmp(trials.dimension(0), trials.dimension(2), trials.dimension(1));
	trca_X1.setZero();
	for (int i = 0; i < trials.dimension(0); i++) {
		trca_X1 = trca_X1 + trials.chip<0>(i);
		trca_X2_tmp.chip<0>(i) = data_->transpose(trials.chip<0>(i));
	}
	//tensor2dToCsv(trca_X1);
	Eigen::Tensor<double, 2> trca_X2 = trca_X2_tmp.chip<0>(0);
	for (int i = 1; i < trca_X2_tmp.dimension(0); ++i) {
		Eigen::Tensor<double, 2> tmp = trca_X2.concatenate(trca_X2_tmp.chip<0>(i), 0);
		trca_X2 = tmp;
	}
	//tensor2dToCsv(trca_X2);

	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	Eigen::Tensor<double, 2> S = trca_X1.contract(data_->transpose(trca_X1), product_dims)
		- data_->transpose(trca_X2).contract(trca_X2, product_dims);
	//tensor2dToCsv(S);
	Eigen::Tensor<double, 2> trca_X2_mean = data_->tensor1to2(trca_X2.mean(Eigen::array<Eigen::DenseIndex, 1>({ 0 })));
	trca_X2 = trca_X2 - trca_X2_mean.broadcast(Eigen::array<int, 2>{ int(trca_X2.dimension(0)), 1});
	Eigen::Tensor<double, 2> Q = data_->transpose(trca_X2).contract(trca_X2, product_dims);
	//tensor2dToCsv(Q);
	Eigen::Tensor<double, 2> eig_vec = data_->solveEig(S, Q);
	return eig_vec;
	
}

Eigen::Tensor<double, 1> TrcaEngine::predict() const  {
	return predict(data_->test_trials_, data_->templates_, U_trca_, U_trca_);
}

Eigen::Tensor<double, 1> TrcaEngine::predict(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates,
	const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const {
	Eigen::Tensor<double, 1> pred_labels(trials.dimension(0));
	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	for (int i = 0; i < trials.dimension(0); i++) {
		Eigen::Tensor<double, 2> r = data_->tensor1to2(filter_banks_weights_).
			contract(canoncorrWithUV(trials.chip<0>(i), templates, U, V), product_dims);
		Eigen::Tensor<double, 0> max_coeff = r.maximum();
		for (int j = 0; j < r.dimension(1); ++j) {
			if (r(j) == max_coeff(0)) {
				pred_labels(i) = j;
			}
		}
	}
	return pred_labels;
}

Eigen::Tensor<double, 2> TrcaEngine::canoncorrWithUV(const Eigen::Tensor<double, 3>& trials, const Eigen::Tensor<double, 4>& templates,
	const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const {
	Eigen::Tensor<double, 2> R(data_->subbands_, data_->stimulus_);
	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	R.setZero();
	for (int i = 0; i < data_->subbands_; ++i) {
		Eigen::Tensor<double, 2> trial = trials.chip<0>(i);

		for (int j = 0; j < data_->stimulus_; ++j) {
			Eigen::Tensor<double, 2> tmplate = templates.chip<0>(j).chip<0>(i);

			Eigen::Tensor<double, 2> A_r = U.chip<0>(i).chip<0>(j);
			Eigen::Tensor<double, 2> B_r = V.chip<0>(i).chip<0>(j);

			Eigen::Tensor<double, 1> a = data_->transpose(A_r).contract(trial, product_dims).chip<0>(0);
			Eigen::Tensor<double, 1> b = data_->transpose(B_r).contract(tmplate, product_dims).chip<0>(0);

			R(i, j) = corrCoef(a,b)(0,1);
		}
	}
	return R;
}

Eigen::Tensor<double, 2> TrcaEngine::corrCoef(Eigen::Tensor<double,1>& x, Eigen::Tensor<double, 1>& y,
	bool rowvar, const std::string& dtype) const{
	// Assuming bias and ddof are deprecated and have no effect
	Eigen::Tensor<double, 2> cov = data_->vecCov(x, y, rowvar, dtype);
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> c(cov.data(), cov.dimension(0), cov.dimension(1));
	Eigen::VectorXd d;
	d = c.diagonal();

	Eigen::VectorXd stddev = d.array().sqrt();
	c = (c.array().rowwise() / stddev.transpose().array()).matrix();
	c = (c.array().colwise() / stddev.array()).matrix();

	// Clip real and imaginary parts to [-1, 1].  This does not guarantee
	// abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
	// excessive work.
	c = c.unaryExpr([](double v) { return std::min(std::max(v, -1.0), 1.0); });

	if (dtype == "complex") {
		// @zikai 23.12.01 for further code.
		// c = c.unaryExpr([](std::complex<double> v) { return std::complex<double>(v.real(), std::min(std::max(v.imag(), -1.0), 1.0)); });
	}

	Eigen::Tensor<double, 2> tensor(c.rows(), c.cols());
	for (int i = 0; i < c.rows(); ++i) {
		for (int j = 0; j < c.cols(); ++j) {
			tensor(i, j) = c(i, j);
		}
	}
	return tensor;
}

