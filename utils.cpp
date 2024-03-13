#include "utils.h"

void tensor1dToCsv(const Eigen::Tensor<double, 1>& tensor, std::string& path) {
    std::ofstream file(path, std::ios::out);
    file.setf(std::ios::fixed, std::ios::floatfield);
    file.precision(12);

    for (int i = 0; i < tensor.dimension(0); i++) {
        file << tensor(i);
        file << ",";
    }

    file.close();
}

// flag: 0(default, ios:out) or 1(append, ios::app)
void tensor2dToCsv(const Eigen::Tensor<double, 2>& tensor, std::string& path, int flag) {
    std::ofstream file(path, std::ios::app);
    if (!flag) {
        file.close();
        file = std::ofstream(path, std::ios::out);
    }
    file.setf(std::ios::fixed, std::ios::floatfield);
    file.precision(12);
        
    for (int i = 0; i < tensor.dimension(0); i++) {
        for (int j = 0; j < tensor.dimension(1); j++) {
            file << tensor(i, j);
            if (j != tensor.dimension(1) - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

void tensor3dToCsv(const Eigen::Tensor<double, 3>& tensor, std::string& path) {
    std::ofstream file(path, std::ios::out);
    file.close();
    for (int i = 0; i < tensor.dimension(0); i++) {
        tensor2dToCsv(tensor.chip<0>(i), path, 1);
    }
}

void tensor4dToCsv(const Eigen::Tensor<double, 4>& tensor, std::string& path) {
    std::ofstream file(path, std::ios::out);
    file.close();
    for (int i = 0; i < tensor.dimension(0); i++) {
        for (int j = 0; j < tensor.dimension(1); j++) {
            tensor2dToCsv(tensor.chip<0>(i).chip<0>(j), path, 1);
        }
    }
}

void WriteToFile(double data) {
    std::ofstream file("./debug.txt", std::ios::app);
    if (file.is_open()) {
        file << data;
        file << "\n";
        file.close();
    }
}



