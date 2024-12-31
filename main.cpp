#include <iostream>
#include <Dense>
#include <Tensor>

int main() {
    // col-major 2x3 tensor
    Eigen::Tensor<double, 2, Eigen::ColMajor> col_major_tensor(2, 3);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
			col_major_tensor(i, j) = i * 3 + j;
		}
	}   
    std::cout << "Col-major tensor:\n" << col_major_tensor << "\n";

    double* data = new double[6];
    memcpy(data, col_major_tensor.data(), col_major_tensor.size() * sizeof(double));
    for (int i = 0; i < 6; i++) {
		std::cout << data[i] << ", ";
	}
    std::cout << std::endl;

    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::ColMajor>> col_major_tensor2(data, 2, 3);
    std::cout << "Col-major tensor2:\n" << col_major_tensor << "\n";

    delete[] data;
    return 0;
}