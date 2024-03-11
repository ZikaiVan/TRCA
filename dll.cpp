#include <Tensor>

extern "C" __declspec(dllexport) void TRCA(double* array, int rows, int cols)
{
    // Map the pointer to an Eigen Tensor for easier manipulation
    Eigen::TensorMap<Eigen::Tensor<double, 2>> data(array, rows, cols);

    // Now you can use 'data' as if it was a regular Eigen::Tensor object
    // For example:
    data(0, 0) = 42.0;  // Set the value at position (0, 0) to 42.0
}