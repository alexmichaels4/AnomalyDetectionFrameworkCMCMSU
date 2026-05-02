#include "Autoencoder.h"
#include <cmath>
#include <random>
#include <stdexcept>

static double randWeight(int fanIn)
{
    static std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0 / fanIn, 1.0 / fanIn);
    return dist(gen);
}

Autoencoder::Autoencoder(int inputSize, int hiddenSize, double learningRate) : in(inputSize), hid(hiddenSize), lr(learningRate)
{
    W1.resize(hid, std::vector<double>(in));
    W2.resize(in, std::vector<double>(hid));
    b1.resize(hid, 0.0);
    b2.resize(in, 0.0);

    for (int i = 0; i < hid; i++)
        for (int j = 0; j < in; j++)
            W1[i][j] = randWeight(in);

    for (int i = 0; i < in; i++)
        for (int j = 0; j < hid; j++)
            W2[i][j] = randWeight(hid);
}

double Autoencoder::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double Autoencoder::dsigmoid(double y)
{
    return y * (1.0 - y);
}

std::vector<double> Autoencoder::forward(const std::vector<double>& x)
{
    if ((int)x.size() != in)
        throw std::runtime_error("Autoencoder: wrong input size");

    std::vector<double> h(hid), y(in);

    // hidden layer
    for (int i = 0; i < hid; i++) {
        double s = b1[i];
        for (int j = 0; j < in; j++)
            s += W1[i][j] * x[j];
        h[i] = sigmoid(s);
    }

    // output layer
    for (int i = 0; i < in; i++) {
        double s = b2[i];
        for (int j = 0; j < hid; j++)
            s += W2[i][j] * h[j];
        y[i] = sigmoid(s);
    }

    return y;
}

double Autoencoder::rmse(const std::vector<double>& x)
{
    auto y = forward(x);
    double err = 0;

    for (int i = 0; i < in; i++) {
        double d = x[i] - y[i];
        err += d * d;
    }

    return std::sqrt(err / in);
}

double Autoencoder::train(const std::vector<double>& x)
{
    if ((int)x.size() != in)
        throw std::runtime_error("Autoencoder: wrong input size");

    std::vector<double> h(hid), y(in);

    // forward
    for (int i = 0; i < hid; i++) {
        double s = b1[i];
        for (int j = 0; j < in; j++)
            s += W1[i][j] * x[j];
        h[i] = sigmoid(s);
    }

    for (int i = 0; i < in; i++) {
        double s = b2[i];
        for (int j = 0; j < hid; j++)
            s += W2[i][j] * h[j];
        y[i] = sigmoid(s);
    }

    // output delta
    std::vector<double> deltaOut(in);
    for (int i = 0; i < in; i++) {
        double e = x[i] - y[i];
        deltaOut[i] = e * dsigmoid(y[i]);
    }

    // hidden delta
    std::vector<double> deltaHid(hid, 0.0);
    for (int i = 0; i < hid; i++) {
        double s = 0;
        for (int j = 0; j < in; j++)
            s += deltaOut[j] * W2[j][i];
        deltaHid[i] = s * dsigmoid(h[i]);
    }

    // update W2, b2
    for (int i = 0; i < in; i++) {
        for (int j = 0; j < hid; j++)
            W2[i][j] += lr * deltaOut[i] * h[j];
        b2[i] += lr * deltaOut[i];
    }

    // update W1, b1
    for (int i = 0; i < hid; i++) {
        for (int j = 0; j < in; j++)
            W1[i][j] += lr * deltaHid[i] * x[j];
        b1[i] += lr * deltaHid[i];
    }

    // return RMSE
    double err = 0;
    for (int i = 0; i < in; i++) {
        double d = x[i] - y[i];
        err += d * d;
    }
    return std::sqrt(err / in);
}

std::vector<double> Autoencoder::featureErrors(const std::vector<double>& x)
{
    auto y = forward(x);
    std::vector<double> errors(in);
    for (int i = 0; i < in; i++)
        errors[i] = std::abs(x[i] - y[i]);
    return errors;
}
