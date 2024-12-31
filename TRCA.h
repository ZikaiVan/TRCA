#ifndef Trca_H
#define Trca_H
#include <Tensor>

class Trca {
public:
	~Trca();
	Trca(int subbands, int stimulus, int electrodes, int num_samples, int train_len=1, int fb_weights_type=0);
	Eigen::Tensor<double, 4> fit(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates);
	Eigen::Tensor<int, 1> predict(const Eigen::Tensor<double, 4>& trials, const Eigen::Tensor<double, 4>& templates,
		const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V, std::vector<double>& coeff) const;

private:
	int subbands_;
	int stimulus_;
	int electrodes_;
	int train_len_;
	int num_samples_;
	int fb_weights_type_;

	Eigen::Tensor<double, 1> filter_banks_weights_;
	Eigen::Tensor<int, 1> possible_classes_;

	Eigen::Tensor<double, 2> trcaU(const Eigen::Tensor<double, 3>& trials) const;
	Eigen::Tensor<double, 2> canoncorrWithUV(const Eigen::Tensor<double, 3>& trials, const Eigen::Tensor<double, 4>& templates,
		const Eigen::Tensor<double, 4>& U, const Eigen::Tensor<double, 4>& V) const;
	Eigen::Tensor<double, 2> corrCoef(Eigen::Tensor<double, 1>& x, Eigen::Tensor<double, 1>& y,
		bool rowvar=true, const std::string& dtype="real") const;
};
#endif //Trca_H