#include "utils.h"

// flag: 0(default, ios:out) or 1(append, ios::app)
void tensor2dToCsv(const Eigen::Tensor<double, 2>& tensor, int flag, const std::string& path) {
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
