#include "FeatureMapper.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

FeatureMapper::FeatureMapper(int featureCount, int maxClusterSize) : n(featureCount), maxClusterSize(maxClusterSize), count(0)
{
    sum.resize(n, 0.0);
    sumSq.resize(n, 0.0);
    cross.resize(n, std::vector<double>(n, 0.0));
}

void FeatureMapper::setFeatureNames(const std::vector<std::string>& names)
{
    featureNames = names;
}

void FeatureMapper::update(const std::vector<double>& x)
{
    if ((int)x.size() != n)
        throw std::runtime_error("FeatureMapper: wrong input size");

    count++;

    for (int i = 0; i < n; i++) {
        sum[i] += x[i];
        sumSq[i] += x[i] * x[i];
    }

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            cross[i][j] += x[i] * x[j];
            if (i != j) cross[j][i] = cross[i][j];
        }
}

std::vector<std::vector<double>> FeatureMapper::computeDistanceMatrix()
{
    std::vector<double> mean(n), stddev(n);
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        mean[i] = sum[i] / count;
        stddev[i] = std::sqrt(sumSq[i] / count - mean[i] * mean[i]);
        if (stddev[i] == 0) stddev[i] = 1e-9;
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double cov = (cross[i][j] / count) - mean[i] * mean[j];
            double corr = cov / (stddev[i] * stddev[j]);
            D[i][j] = 1.0 - corr;
        }

    return D;
}

void FeatureMapper::finalize()
{
    auto D = computeDistanceMatrix();
    hierarchicalClustering(D);
}

void FeatureMapper::hierarchicalClustering(const std::vector<std::vector<double>>& D)
{
    std::cout << "Forming clusters, please wait a few minutes..." << std::endl;
    clusters.clear();

    for (int i = 0; i < n; i++)
        clusters.push_back({ i });

    while (true) {
        double bestDist = std::numeric_limits<double>::max();
        int a = -1, b = -1;

        for (int i = 0; i < (int)clusters.size(); i++)
            for (int j = i + 1; j < (int)clusters.size(); j++) {
                if ((int)(clusters[i].size() + clusters[j].size()) > maxClusterSize)
                    continue;

                double dist = 0;
                int cnt = 0;

                for (int x : clusters[i])
                    for (int y : clusters[j]) {
                        dist += D[x][y];
                        cnt++;
                    }

                dist /= cnt;

                if (dist < bestDist) {
                    bestDist = dist;
                    a = i;
                    b = j;
                }
            }

        if (a == -1) break;

        clusters[a].insert(clusters[a].end(),
            clusters[b].begin(), clusters[b].end());
        clusters.erase(clusters.begin() + b);
    }

    /*
    std::cout << "\nFeature clusters:\n\n";
    for (auto& c : clusters) {
        std::cout << "[ ";
        for (int i : c) {
            if (!featureNames.empty())
                std::cout << featureNames[i] << " ";
            else
                std::cout << i << " ";
        }
        std::cout << "]\n\n";
    }
    */
}

const std::vector<std::vector<int>>& FeatureMapper::getClusters() const
{
    return clusters;
}

const std::vector<std::string>& FeatureMapper::getFeatureNames() const
{
    return featureNames;
}

void FeatureMapper::resetStats()
{
    count = 0;
    std::fill(sum.begin(), sum.end(), 0.0);
    std::fill(sumSq.begin(), sumSq.end(), 0.0);
    for (auto& row : cross)
        std::fill(row.begin(), row.end(), 0.0);
}

void FeatureMapper::updateIfNormal(const std::vector<double>& x, bool isNormal)
{
    if (!isNormal) return; //ignore anomalies
    update(x);
    sinceLastCluster++;
}

bool FeatureMapper::shouldRecluster()
{
    return sinceLastCluster >= reclusterInterval;
}
