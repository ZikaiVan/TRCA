#include "TRCA.h"
#include "utils.h"
#include <iostream>

Trca::~Trca() {}

Trca::Trca(int subbands, int stimulus, int electrodes, int num_samples, int train_len, int fb_weights_type) {
	subbands_ = subbands;
	stimulus_ = stimulus;
	electrodes_ = electrodes;
	train_len_  = train_len;
	num_samples_ = num_samples;
	fb_weights_type_ = fb_weights_type;

	Eigen::Tensor<double, 1> filter_banks_weights(subbands_);
	Eigen::Tensor<int, 1> possible_classes(stimulus_);
	for (int i = 0; i < stimulus_; i++) {
		possible_classes(i) = i;
	}
	possible_classes_ = possible_classes;
	if (fb_weights_type_ == 1) {
		for (int i = 0; i < subbands_; i++) {
			filter_banks_weights(i) = pow(i+1, -1.75) + 0.5;
		}
	}
	else {
		for (int i = 0; i < subbands_; i++) {
			filter_banks_weights(i) = pow(i+1, -1.25) + 0.15;
		}
	}
	filter_banks_weights_ = filter_banks_weights;
}

Eigen::Tensor<double, 4> Trca::fit(
	const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates) 
{
	// @zikai 23.11.29 Here we fixed component = 1.
	int component = 1;
	Eigen::Tensor<double, 4> U_trca(subbands_, stimulus_, electrodes_, component);
	for (int i = 0; i < subbands_; ++i) {
		Eigen::Tensor<double, 4> trains(stimulus_, train_len_, electrodes_, num_samples_);
		for (int j = 0; j < stimulus_; j++) {
			int m = 0; // calculate train blocks
			for (int k = j; k < trials.dimension(0); k += stimulus_, m++) {
				trains.chip<0>(j).chip<0>(m) = trials.chip<0>(k).chip<0>(i);
			}
		}
		Eigen::Tensor<double, 3> U(trains.dimension(0), electrodes_, electrodes_);
		for (int j = 0; j < trains.dimension(0); j++) {
			// @zikai 23.11.29: different from MATLAB in positions and multiples.
			U.chip<0>(j) = trcaU(trains.chip<0>(j));
		}
		for (int j = 0; j < stimulus_; j++) {
			U_trca.chip<0>(i).chip<0>(j) = U.chip<0>(j)
				.slice(Eigen::array<int, 2>({ 0, 0 }), Eigen::array<int, 2>({ electrodes_, component }));
		}
	}
	// @zikai due to the differece, maybe some wrong result will generate, hasn't checked.
	return U_trca;
}

Eigen::Tensor<double, 2> Trca::trcaU(const Eigen::Tensor<double, 3>& trials) const {
	Eigen::Tensor<double, 2> trca_X1(trials.dimension(1), trials.dimension(2));
	Eigen::Tensor<double, 3> trca_X2_tmp(trials.dimension(0), trials.dimension(2), trials.dimension(1));
	trca_X1.setZero();
	for (int i = 0; i < trials.dimension(0); i++) {
		trca_X1 = trca_X1 + trials.chip<0>(i);
		trca_X2_tmp.chip<0>(i) = transpose(trials.chip<0>(i));
	}

	Eigen::Tensor<double, 2> trca_X2 = trca_X2_tmp.chip<0>(0);
	for (int i = 1; i < trca_X2_tmp.dimension(0); ++i) {
		Eigen::Tensor<double, 2> tmp = trca_X2.concatenate(trca_X2_tmp.chip<0>(i), 0);
		trca_X2 = tmp;
	}

	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	Eigen::Tensor<double, 2> S = trca_X1.contract(transpose(trca_X1), product_dims)
		- transpose(trca_X2).contract(trca_X2, product_dims);
	Eigen::Tensor<double, 2> trca_X2_mean = tensor1to2(trca_X2.mean(Eigen::array<Eigen::DenseIndex, 1>({ 0 })));
	trca_X2 = trca_X2 - trca_X2_mean.broadcast(Eigen::array<int, 2>{ int(trca_X2.dimension(0)), 1});
	Eigen::Tensor<double, 2> Q = transpose(trca_X2).contract(trca_X2, product_dims);
	Eigen::Tensor<double, 2> eig_vec = solveEig(S, Q);

	return eig_vec;
}

Eigen::Tensor<int, 1> Trca::predict(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates,
	const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const {
	Eigen::Tensor<int, 1> pred_labels(trials.dimension(0));
	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	for (int i = 0; i < trials.dimension(0); i++) {
		Eigen::Tensor<double, 2> r = tensor1to2(filter_banks_weights_).
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

Eigen::Tensor<double, 2> Trca::canoncorrWithUV(const Eigen::Tensor<double, 3>& trials, const Eigen::Tensor<double, 4>& templates,
	const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const {
	Eigen::Tensor<double, 2> R(subbands_, stimulus_);
	Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
	R.setZero();
	for (int i = 0; i < subbands_; ++i) {
		Eigen::Tensor<double, 2> trial = trials.chip<0>(i);

		for (int j = 0; j < stimulus_; ++j) {
			Eigen::Tensor<double, 2> tmplate = templates.chip<0>(j).chip<0>(i);

			Eigen::Tensor<double, 2> A_r = U.chip<0>(i).chip<0>(j);
			Eigen::Tensor<double, 2> B_r = V.chip<0>(i).chip<0>(j);

			Eigen::Tensor<double, 1> a = transpose(A_r).contract(trial, product_dims).chip<0>(0);
			Eigen::Tensor<double, 1> b = transpose(B_r).contract(tmplate, product_dims).chip<0>(0);

			R(i, j) = corrCoef(a,b)(0,1);
		}
	}
	return R;
}

Eigen::Tensor<double, 2> Trca::corrCoef(Eigen::Tensor<double,1>& x, Eigen::Tensor<double, 1>& y,
	bool rowvar, const std::string& dtype) const{
	// Assuming bias and ddof are deprecated and have no effect
	Eigen::Tensor<double, 2> cov = vecCov(x, y, rowvar, dtype);
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

