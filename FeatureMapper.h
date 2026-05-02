#pragma once
#include <vector>
#include <string>

class FeatureMapper {
public:
    FeatureMapper(int featureCount, int maxClusterSize);

    void update(const std::vector<double>& x);
    void finalize();

    const std::vector<std::vector<int>>& getClusters() const;
    const std::vector<std::string>& getFeatureNames() const;
    void setFeatureNames(const std::vector<std::string>& names);

    void resetStats();
    void updateIfNormal(const std::vector<double>& x, bool isNormal);

    int reclusterInterval = 500;
    int sinceLastCluster = 0;

    bool shouldRecluster();


private:
    int n;
    int maxClusterSize;
    long count;

    std::vector<double> sum;
    std::vector<double> sumSq;
    std::vector<std::vector<double>> cross;

    std::vector<std::vector<int>> clusters;
    std::vector<std::string> featureNames;

    std::vector<std::vector<double>> computeDistanceMatrix();
    void hierarchicalClustering(const std::vector<std::vector<double>>& D);
};
