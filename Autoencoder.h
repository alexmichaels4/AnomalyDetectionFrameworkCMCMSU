#pragma once
#include <vector>

class Autoencoder {
public:
    Autoencoder(int inputSize, int hiddenSize, double learningRate = 0.01);

    std::vector<double> forward(const std::vector<double>& x);
    double train(const std::vector<double>& x);
    double rmse(const std::vector<double>& x);
    std::vector<double> featureErrors(const std::vector<double>& x);

private:
    int in, hid;
    double lr;

    std::vector<std::vector<double>> W1, W2;
    std::vector<double> b1, b2;

    static double sigmoid(double x);
    static double dsigmoid(double y);
};
