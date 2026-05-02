#pragma once
#include <vector>
#include "Autoencoder.h"

struct FeatureContribution
{
    int featureIndex;
    int clusterIndex;
    double error;
};

class AnomalyDetector {
public:
    AnomalyDetector(const std::vector<std::vector<int>>& clusters, double learningRate = 0.01);

    double train(const std::vector<double>& x);
    double score(const std::vector<double>& x);

    double getPhi() const;
    bool isAnomaly(double score) const;

    void rebuild(const std::vector<std::vector<int>>& newClusters);

    std::vector<FeatureContribution> explain(const std::vector<double>& x, int topN = 3); // top-N features with biggest contribution to the anomaly

private:
    std::vector<std::vector<int>> clusters;
    std::vector<Autoencoder> ensemble;
    Autoencoder outputAE;

    double phi;

    std::vector<double> extract(const std::vector<double>& x, const std::vector<int>& idx);
};
