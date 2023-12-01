#include "PreprocessEngine.h"
#include <cmath>
#include <iostream>
#define M_PI 3.1415926

PreprocessEngine::~PreprocessEngine() {}

PreprocessEngine::PreprocessEngine(const SSVEP* data) {
	s_rate_ = data->s_rate_;
	this->data_ = data;
	// 写死参数，可以改为config.ini配置.
	// Lowpass, Highpass, bandPass, bandStop
	bsf_ = new Cheby1Filter(4, 2, 47, 53, s_rate_, 's');
	bpf_ = new Cheby1Filter[data->subbands_];
	for (int i = 0; i < data->subbands_; i++) {
		bpf_[i] = Cheby1Filter(4, 1, 9*(i+1), 90, s_rate_, 'p');
	}
}

Eigen::Tensor<double, 2> PreprocessEngine::notch(const Eigen::Tensor<double, 2>& trial) {
	return filtFilt(trial, *bsf_, 1);
}

Eigen::Tensor<double, 3> PreprocessEngine::filterBank(const Eigen::Tensor<double, 2>& trial) {
	Eigen::array<Eigen::DenseIndex, 3> offsets = { 0, 0, data_->latency_ };
	Eigen::array<Eigen::DenseIndex, 3> extents = { data_->subbands_, data_->electrodes_, data_->duration_ };
	Eigen::Tensor<double, 3> trial_tmp(data_->subbands_, trial.dimension(0), trial.dimension(1));
	Eigen::Tensor<double, 3> trial_sets(data_->subbands_, trial.dimension(0), data_->duration_);
	for (int i = 0; i < data_->subbands_; i++) {
		trial_tmp.chip<0>(i)=filtFilt(trial, bpf_[i], 1);
	}
	trial_sets = trial_tmp.slice(offsets, extents);
	//tensor3dToCsv(trial_sets);
	return trial_sets;
}

Eigen::Tensor<double, 2> PreprocessEngine::filtFilt(const Eigen::Tensor<double, 2>& trial, const Cheby1Filter& filter, int axis) {
	int edge = 3 * (std::max(filter.b_.size(), filter.a_.size()) - 1);
	Eigen::Tensor<double, 2> ext = oddExt(trial, edge, axis);
	Eigen::Tensor<double, 2> zi = lFilterZi(filter);

	// Forward filter.
	Eigen::Tensor<double, 2> trial_0 = axisSlice(ext, 0, 1, 1, 1);
	Eigen::Tensor<double, 2> ext_1 = lFilter(ext,
		zi.broadcast(Eigen::array<int, 2>({ int(trial_0.dimension(0)), 1 }))
		* trial_0.broadcast(Eigen::array<int, 2>({ 1, int(zi.dimension(1)) })), filter);
	//tensor2dToCsv(trial_1);

	// Backward filter.
	trial_0 = axisSlice(ext_1, ext_1.dimension(1) - 1, ext_1.dimension(1), -1, axis);
	ext_1 = axisSlice(ext_1, 0, ext_1.dimension(1), -1, axis);
	Eigen::Tensor<double, 2> ext_2 = lFilter(ext_1,
		zi.broadcast(Eigen::array<int, 2>({ int(trial_0.dimension(0)), 1 }))
		* trial_0.broadcast(Eigen::array<int, 2>({ 1, int(zi.dimension(1)) })), filter);
	//tensor2dToCsv(trial_2);

	// Reverse ext_2.
	ext_2 = axisSlice(ext_2, 0, ext_2.dimension(1), -1, axis);
	if (edge > 0) {
		// Slice the actual signal from the extended signal.
		return axisSlice(ext_2, edge, ext_2.dimension(1) - edge, 1, axis);
	}
	return ext_2;
}

Eigen::Tensor<double, 2> PreprocessEngine::oddExt(const Eigen::Tensor<double, 2>& x, int edge, int axis) {
	Eigen::Tensor<double, 2> left_ext = axisSlice(x, 1, edge + 1, -1, axis);
	Eigen::Tensor<double, 2> left_end = axisSlice(x, 0, 1, 1, axis).broadcast(Eigen::array<int, 2>({ 1, int(left_ext.dimension(1)) }));
	Eigen::Tensor<double, 2> right_ext = axisSlice(x, x.dimension(1) - (edge + 1), x.dimension(1) - 1, -1, axis);
	Eigen::Tensor<double, 2> right_end = axisSlice(x, x.dimension(1) - 1, x.dimension(1), 1, axis).broadcast(Eigen::array<int, 2>({ 1, int(right_ext.dimension(1)) }));
	Eigen::Tensor<double, 2> ext = (2 * left_end - left_ext).concatenate(x, axis).concatenate(2 * right_end - right_ext, axis);
	//std::cout << ext;
	return ext;
}

Eigen::Tensor<double, 2> PreprocessEngine::axisSlice(const Eigen::Tensor<double, 2>& a, int start, int stop, int step, int axis) {
	Eigen::array<Eigen::Index, 2> dims = a.dimensions();
	Eigen::array<Eigen::Index, 2> start_indices = { 0, start };
	Eigen::array<Eigen::Index, 2> offset_indices = { dims[0], stop - start };
	Eigen::array<Eigen::Index, 2> strides = { 1, 1 };

	Eigen::Tensor<double, 2> b = a.slice(start_indices, offset_indices).stride(strides);
	if (step < 0) {
		Eigen::Tensor<double, 2> c = b.reverse(Eigen::array<bool, 2>({ false, true }));
		b = c;
	}
	return b;
}

Eigen::Tensor<double, 2> PreprocessEngine::lFilterZi(const Cheby1Filter& filter) {
	Eigen::Tensor<double, 1> b_copy = filter.b_;
	Eigen::Tensor<double, 1> a_copy = filter.a_;

	while (a_copy.dimension(0) > 1 && a_copy(0) == 0.0) {
		Eigen::array<Eigen::Index, 1> start_indices = { 1 };
		Eigen::array<Eigen::Index, 1> offset_indices = { a_copy.dimension(0) };
		a_copy = a_copy.slice(start_indices, offset_indices);
	}

	if (a_copy.dimension(0) < 1) {
		throw std::invalid_argument("There must be at least one nonzero `a` coefficient.");
	}

	int n = std::max(a_copy.size(), b_copy.size());
	Eigen::Tensor<double, 2> a_copy2D = data_->tensor1to2(a_copy);
	Eigen::Tensor<double, 2> I = data_->identity(n - 1);
	Eigen::Tensor<double, 2> A = data_->transpose(data_->rowCompanion(a_copy2D));
	Eigen::Tensor<double, 2> IminusA = I - A;

	Eigen::Tensor<double, 2> b_copy2D_T = data_->transpose(data_->tensor1to2(b_copy));
	Eigen::Tensor<double, 2> a_copy2D_T = data_->transpose(data_->tensor1to2(a_copy));
	Eigen::array<Eigen::Index, 2> start_indices = { 1,0 };
	Eigen::array<Eigen::Index, 2> offset_indices = { b_copy2D_T.dimension(0) - 1, 1 };
	Eigen::Tensor<double, 2> B = b_copy2D_T.slice(start_indices, offset_indices)
		- a_copy2D_T.slice(start_indices, offset_indices) * b_copy2D_T(0, 0);

	//Solve A*zi = B
	Eigen::Tensor<double, 2> zi = data_->solveZi(IminusA, B);
	//std::cout << zi;
	return zi;
}

Eigen::Tensor<double, 2> PreprocessEngine::lFilter(const Eigen::Tensor<double, 2>& x, Eigen::Tensor<double, 2> z, const Cheby1Filter& filter) {
	Eigen::Tensor<double, 2> y(x.dimension(0), x.dimension(1));
	y.setZero();
	Eigen::Tensor<double, 1> b_copy = filter.b_;
	Eigen::Tensor<double, 1> a_copy = filter.a_;
	Eigen::Tensor<double, 2> z_copy = z;
	int order = std::max(a_copy.dimension(0), b_copy.dimension(0));

	while (a_copy.dimension(0) > 1 && a_copy(0) == 0.0) {
		Eigen::array<Eigen::Index, 1> start_indices = { 1 };
		Eigen::array<Eigen::Index, 1> offset_indices = { a_copy.dimension(0) };
		a_copy = a_copy.slice(start_indices, offset_indices);
	}
	if (a_copy.dimension(0) < 1) {
		throw std::invalid_argument("There must be at least one nonzero `a` coefficient.");
	}
	if (z_copy.dimension(1) < order) {
		Eigen::Tensor<double, 2> zeros(z_copy.dimension(0), order - z_copy.dimension(1));
		Eigen::Tensor<double, 2> zz_copy = z_copy.concatenate(zeros.setZero(), 1);
		z_copy = zz_copy;
	}

	for (int i = 0; i < x.dimension(0); i++) {
		for (int j = 0; j < x.dimension(1); ++j) {
			int k = order - 1;
			while (k)
			{
				if (j >= k) {
					z_copy(i, k - 1) = b_copy(k) * x(i, j - k) - a_copy(k) * y(i, j - k) + z_copy(i, k);
				}
				--k;
			}
			y(i, j) = b_copy(0) * x(i, j) + z_copy(i, 0);
		}
	}
	return y;
}