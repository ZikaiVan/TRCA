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

Eigen::Tensor<double, 2> SSVEP::getSingleTrial(int block, int stimulus, int start, int end) {
    Eigen::array<Eigen::DenseIndex, 2> offsets = { 0, start};
    Eigen::array<Eigen::DenseIndex, 2> extents = { electrodes_, end-start};
    Eigen::Tensor<double, 2> slice_data = this->data.chip(block, 0).chip(stimulus, 0).slice(offsets, extents);
    //std::cout << std::endl<<slice_data.dimensions() << std::endl;
    return slice_data;
}

// private
void SSVEP::loadConfig(const std::string& path)
{
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file");
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (line.find("s_rate") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->s_rate_ = std::stoi(value);
        }
        else if (line.find("latency") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->latency_ = std::stod(value);
        }
        else if (line.find("duration") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->duration_ = std::stod(value);
        }
        else if (line.find("pre_stim") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->pre_stim_ = std::stod(value);
        }
        else if (line.find("rest") != std::string::npos) {
            std::string value = line.substr(line.find("=") + 1);
            this->rest_ = std::stod(value);
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
    samples_ = (latency_ + duration_ + pre_stim_ + rest_) * s_rate_;
}
