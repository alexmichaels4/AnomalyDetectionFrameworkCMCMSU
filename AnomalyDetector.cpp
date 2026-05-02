#include "AnomalyDetector.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

static int hiddenSize(int n)
{
    return std::max(1, (int)std::ceil(0.75 * n));
}

AnomalyDetector::AnomalyDetector(const std::vector<std::vector<int>>& clusters, double learningRate) : clusters(clusters), phi(0.0), outputAE((int)clusters.size(), hiddenSize((int)clusters.size()), learningRate)
{
    for (auto& c : clusters) {
        ensemble.emplace_back(c.size(), hiddenSize(c.size()), learningRate);
    }
}

std::vector<double> AnomalyDetector::extract(const std::vector<double>& x, const std::vector<int>& idx)
{
    std::vector<double> v;
    v.reserve(idx.size());
    for (int i : idx)
        v.push_back(x[i]);
    return v;
}

double AnomalyDetector::train(const std::vector<double>& x)
{
    if (clusters.empty())
        throw std::runtime_error("AnomalyDetector not initialized");

    std::vector<double> rmseVec;

    // training ensemble layer
    for (size_t i = 0; i < clusters.size(); i++) {
        auto v = extract(x, clusters[i]);
        double err = ensemble[i].train(v);
        rmseVec.push_back(err);
    }

    double outErr = outputAE.train(rmseVec);

    // threshold
    if (outErr > phi)
        phi = outErr;

    return outErr;
}

double AnomalyDetector::score(const std::vector<double>& x)
{
    std::vector<double> rmseVec;

    for (size_t i = 0; i < clusters.size(); i++) {
        auto v = extract(x, clusters[i]);
        rmseVec.push_back(ensemble[i].rmse(v));
    }

    return outputAE.rmse(rmseVec);
}

double AnomalyDetector::getPhi() const
{
    return phi;
}

bool AnomalyDetector::isAnomaly(double s) const
{
    return s > phi;
}

void AnomalyDetector::rebuild(const std::vector<std::vector<int>>& newClusters)
{
    clusters = newClusters;
    ensemble.clear();

    for (auto& c : clusters) {
        ensemble.emplace_back(c.size(), hiddenSize(c.size()));
    }

    outputAE = Autoencoder((int)clusters.size(), hiddenSize((int)clusters.size()));
    phi = 0.0;
}

std::vector<FeatureContribution> AnomalyDetector::explain(const std::vector<double>& x, int topN)
{
    std::vector<FeatureContribution> all;

    for (size_t ci = 0; ci < clusters.size(); ci++) {
        auto sub = extract(x, clusters[ci]);

        auto errs = ensemble[ci].featureErrors(sub);

        for (size_t fi = 0; fi < clusters[ci].size(); fi++) {
            all.push_back({clusters[ci][fi], (int)ci, errs[fi]});
        }
    }

    std::sort(all.begin(), all.end(),
        [](const FeatureContribution& a, const FeatureContribution& b) {
            return a.error > b.error;
        });

    if ((int)all.size() > topN)
        all.resize(topN);

    return all;
}