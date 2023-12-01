#include "SSVEP.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"

// public
SSVEP::SSVEP(const std::string& path) {
    loadConfig(path);
    Eigen::Tensor<int, 1> train_labels(stimulus_ * train_len_);
    Eigen::Tensor<double, 4> train_trials(train_len_ * stimulus_, subbands_, electrodes_, duration_);
    for (int block = 0; block < train_len_; block++) {
        for (int stimulus = 0; stimulus < stimulus_; stimulus++) {
            train_labels(block * stimulus_ + stimulus) = stimulus;
        }
    }
    train_labels_ = train_labels;
    train_trials_ = train_trials;

    Eigen::Tensor<int, 1> test_labels(stimulus_ * test_len_);
    Eigen::Tensor<double, 4> test_trials(test_len_ * stimulus_, subbands_, electrodes_, duration_);
    for (int block = train_len_; block < blocks_; block++){
        for (int stimulus = 0; stimulus < stimulus_; stimulus++) {
            test_labels((block-train_len_) * stimulus_ + stimulus) = stimulus;
        }
    }
    test_labels_ = test_labels;
    test_trials_ = test_trials;
}

void SSVEP::loadMat(const std::string& path){}

void SSVEP::loadCsv(const std::string& path)
{
    int rows = 0, cols = 0;
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    std::string line;
    std::vector<std::vector<double>> values;
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ','))
        {
            double val = std::stod(cell);
            row.push_back(val);
        }
        values.push_back(row);// (electrodes, samples, set, block, stimulus)
        ++rows;
        cols = row.size() / (sets_ + 1);
    }

    Eigen::Tensor<double, 4> data(blocks_, stimulus_, electrodes_, samples_);
    data.setZero();
    for (int i = 0; i < electrodes_; i++) {
        for (int j = 0; j < samples_; j++) {
            for (int k = 0; k < blocks_; k++) {
                for (int u = 0; u < stimulus_; u++) {
                    data(k, u, i, j) =
                        values[i][j + sets_ * samples_ + k * samples_ * (sets_+1) + u * samples_ * (sets_ + 1) * blocks_];
                    //std::cout << data(k, u, i, j) << std::endl;
                }
			}
        }
    }
    data_ = data;
    //std::cout << std::endl << data.dimensions() << std::endl;
}

void SSVEP::calculateTemplates() {
    const Eigen::Tensor<double, 4> tensor = train_trials_;
    Eigen::Tensor<double, 4> templates(stimulus_, tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
    for (int i = 0; i < stimulus_; ++i) {
        int k = 0;
        Eigen::Tensor<double, 4> template_trials(train_len_, tensor.dimension(1), tensor.dimension(2), tensor.dimension(3));
        for (int j = i; j < tensor.dimension(0); j += stimulus_, k++) {
            template_trials.chip<0>(k) = tensor.chip<0>(j);
        }
        templates.chip<0>(i) = template_trials.mean(Eigen::array<Eigen::DenseIndex, 1>({ 0 }));
    }
    templates_ = templates;
}

Eigen::Tensor<double, 2> SSVEP::getSingleTrial(int block, int stimulus) const {
    Eigen::array<Eigen::DenseIndex, 2> offsets = {0, pre_stim_};
    Eigen::array<Eigen::DenseIndex, 2> extents = {electrodes_, latency_+duration_};
    Eigen::Tensor<double, 2> single_trial = data_.chip(block, 0).chip(stimulus, 0).slice(offsets, extents);
    //std::cout << std::endl<<slice_data.dimensions() << std::endl;
    return single_trial;
}

Eigen::Tensor<double, 2> SSVEP::tensor1to2(const Eigen::Tensor<double, 1>& tensor) const {
    Eigen::Tensor<double, 2> tensor1(1, tensor.dimension(0));
    for (int i = 0; i < tensor.dimension(0); i++) {
        tensor1(0, i) = tensor(i);
    }
    return tensor1;
}

Eigen::Tensor<double, 2> SSVEP::transpose(const Eigen::Tensor<double, 2>& tensor) const {
    Eigen::array<int, 2> shuffleOrder({ 1, 0 });
    return tensor.shuffle(shuffleOrder);
}

Eigen::Tensor<double, 2> SSVEP::rowCompanion(const Eigen::Tensor<double, 2>& tensor) const {
    int n = tensor.dimension(1);
    Eigen::Tensor<double, 2> companion(n - 1, n - 1);
    companion.setZero();
    for (int i = 0; i < n - 2; i++) {
        companion(i + 1, i) = 1.0;
    }
    for (int i = 0; i < n-1; i++) {
        companion(0, i) = -tensor(0, i+1) / tensor(0, 0);
    }
    return companion;
}

Eigen::Tensor<double, 2> SSVEP::identity(int n) const {
    Eigen::Tensor<double, 2> I(n, n);
    I.setZero();
    for (int i = 0; i < n; i++) {
        I(i, i) = 1.0;
    }
    return I;
}

Eigen::Tensor<double, 2> SSVEP::solveZi(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B) const {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> A_(A.data(), A.dimension(0), A.dimension(1));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> B_(B.data(), B.dimension(0), B.dimension(1));
    Eigen::VectorXd zi = A_.fullPivLu().solve(B_);

    Eigen::Tensor<double, 2> tensor(1, zi.size());
    for (int i = 0; i < zi.size(); i++) {
        tensor(0, i) = zi(i);
    }
    return tensor;
}

Eigen::Tensor<double, 2> SSVEP::solveEig(const Eigen::Tensor<double, 2>& S, const Eigen::Tensor<double, 2>& Q) const {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> S_(S.data(), S.dimension(0), S.dimension(1));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> Q_(Q.data(), Q.dimension(0), Q.dimension(1));
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> solver;
    solver.compute(S_, Q_);
    //Eigen::VectorXcd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd realEigenvectors = solver.eigenvectors().real();
    Eigen::Tensor<double, 2> tensor(realEigenvectors.rows(), realEigenvectors.cols());
    for (int i = 0; i < realEigenvectors.rows(); ++i) {
        for (int j = 0; j < realEigenvectors.cols(); ++j) {
            tensor(i, j) = realEigenvectors(i, j);
        }
    }
    return tensor;
}

Eigen::Tensor<double, 2> SSVEP::vecCov(Eigen::Tensor<double, 1>& a, Eigen::Tensor<double, 1>& b,
    bool rowvar, const std::string& dtype) const {
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(a.data(), a.size());
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(b.data(), b.size());
    Eigen::MatrixXd x_mean = x.array() - x.mean();
    Eigen::MatrixXd y_mean = y.array() - y.mean();
    Eigen::Tensor<double, 2> c(2, 2);
    double normalizer = static_cast<double>(x.size() - 1);

    c(0, 0) = (x_mean.array() * x_mean.array()).sum() / normalizer;
    c(0, 1) = (x_mean.array() * y_mean.array()).sum() / normalizer;
    c(1, 0) = c(0, 1);
    c(1, 1) = (y_mean.array() * y_mean.array()).sum() / normalizer;
    return c;
}

// private
void SSVEP::loadConfig(const std::string& path)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("s_rate") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->s_rate_ = std::stoi(value);
            break;
        }
    }
    while (std::getline(in, line)) {
        if (line.find("train_blocks") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->train_len_ = std::stod(value);
        }
        else if (line.find("latency") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->latency_ = std::stod(value) *this->s_rate_;
        }
        else if (line.find("duration") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->duration_ = std::stod(value) * this->s_rate_;
        }
        else if (line.find("pre_stim") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->pre_stim_ = std::stod(value) * this->s_rate_;
        }
        else if (line.find("rest") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->rest_ = std::stod(value) * this->s_rate_;
        }
        else if (line.find("subbands") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->subbands_ = std::stoi(value) > 9 ? 9 : std::stoi(value);
            this->subbands_ = std::stoi(value) < 1 ? 1 : std::stoi(value);
        }
        else if (line.find("type") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->type_ = std::stoi(value);
        }
        else if (line.find("electrodes") != std::string::npos) {
			std::string value = line.substr(line.find("=") + 1);
			this->electrodes_ = std::stoi(value);
		}
        else if (line.find("stimulus") != std::string::npos) {
			std::string value = line.substr(line.find("=") + 1);
			this->stimulus_ = std::stoi(value);
		}
        else if (line.find("blocks") != std::string::npos) {
			std::string value = line.substr(line.find("=") + 1);
			this->blocks_ = std::stoi(value);
		}
        else if (line.find("sets") != std::string::npos) {
			std::string value = line.substr(line.find("=") + 1);
			this->sets_ = std::stoi(value);
		}
    }
    test_len_ = blocks_ - train_len_;
    samples_ = latency_ + duration_ + pre_stim_ + rest_;
}
