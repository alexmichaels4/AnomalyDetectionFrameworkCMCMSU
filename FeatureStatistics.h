#pragma once
#include <vector>
#include <string>
#include <limits>

struct FeatureStats {
    double mean = 0.0;
    double stddev = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
};

class FeatureStatistics {
public:
    FeatureStatistics(int featureCount, const std::vector<std::string>& names);

    void update(const std::vector<double>& x); // during training
    void finalize(); // after training

    bool isSignificantDeviation(int idx, double value, double zThreshold = 2.5) const;
    double getZScore(int idx, double value) const;
    std::string getName(int idx) const;
    const FeatureStats& getStats(int idx) const;

    // report building blocks
    std::string getReconstructionLevel(double contributionPct) const;
    std::vector<std::string> getInterpretationBullets(int idx, double value) const;
    std::vector<std::string> getPossibleCausesBullets(int idx) const;
    std::string getRiskLevel(const std::vector<std::pair<int, double>>& featureZScores) const;
    std::vector<std::string> getMostProbableScenarios(const std::vector<int>& featureIndices) const;
    std::vector<std::string> getRecommendedActions(const std::vector<int>& featureIndices) const;

private:
    int n;
    std::vector<std::string> names;
    std::vector<FeatureStats> stats;
    std::vector<double> sums;
    std::vector<double> sumSqs;
    std::vector<long> counts;

    enum class FeatureCategory {
        LOGIN, RESPONSE_TIME, FILES_PROCESSES, TRAFFIC_PACKETS,
        DNS, PORT, TIME_PATTERN, PERMISSION, LOG, ANTIVIRUS,
        SYSTEM_CONFIG, SOFTWARE, TERMINATION, TLS, SAME_IP, UNKNOWN
    };

    FeatureCategory getCategory(int idx) const;
    std::string toLower(const std::string& s) const;
};