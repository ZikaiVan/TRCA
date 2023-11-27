#include "SSVEP.h"
#include <iostream>
#include <fstream>
#include <vector>

// public
SSVEP::SSVEP(const std::string& path) {
    loadConfig(path);
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
    //std::cout << std::endl << data.dimensions() << std::endl;
    this->data = data;
}

Eigen::Tensor<double, 2> SSVEP::getSingleTrial(int block, int stimulus) {
    Eigen::array<Eigen::DenseIndex, 2> offsets = {0, pre_stim_};
    Eigen::array<Eigen::DenseIndex, 2> extents = {electrodes_, latency_+duration_};
    Eigen::Tensor<double, 2> single_trial = data.chip(block, 0).chip(stimulus, 0).slice(offsets, extents);
    //std::cout << std::endl<<slice_data.dimensions() << std::endl;
    return single_trial;
}

Eigen::Tensor<double, 2> SSVEP::tensor1to2(const Eigen::Tensor<double, 1>& tensor1) {
    Eigen::Tensor<double, 2> tensor2(1, tensor1.dimension(0));
    for (int i = 0; i < tensor1.dimension(0); i++) {
        tensor2(0, i) = tensor1(i);
    }
    return tensor2;
}

Eigen::Tensor<double, 2> SSVEP::transpose(const Eigen::Tensor<double, 2>& tensor) {
    Eigen::array<int, 2> shuffleOrder({ 1, 0 });
    return tensor.shuffle(shuffleOrder);
}

Eigen::Tensor<double, 2> SSVEP::rowCompanion(const Eigen::Tensor<double, 2>& input) {
    int n = input.dimension(1);
    Eigen::Tensor<double, 2> tensor(n - 1, n - 1);
    tensor.setZero();
    for (int i = 0; i < n - 2; i++) {
        tensor(i + 1, i) = 1.0;
    }
    for (int i = 0; i < n-1; i++) {
        tensor(0, i) = -input(0, i+1) / input(0, 0);
    }
    return tensor;
}

Eigen::Tensor<double, 2> SSVEP::identity(int n) {
    Eigen::Tensor<double, 2> I(n, n);
    I.setZero();
    for (int i = 0; i < n; i++) {
        I(i, i) = 1.0;
    }
    return I;
}

Eigen::Tensor<double, 2> SSVEP::solve(const Eigen::Tensor<double, 2>& A, const Eigen::Tensor<double, 2>& B) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> A_(A.data(), A.dimension(0), A.dimension(1));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> B_(B.data(), B.dimension(0), B.dimension(1));
    Eigen::VectorXd zi = A_.fullPivLu().solve(B_);

    Eigen::Tensor<double, 2> tensor(1, zi.size());
    for (int i = 0; i < zi.size(); i++) {
        tensor(0, i) = zi(i);
    }
    return tensor;
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
        if (line.find("latency") != std::string::npos) {
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
    samples_ = latency_ + duration_ + pre_stim_ + rest_;
}
