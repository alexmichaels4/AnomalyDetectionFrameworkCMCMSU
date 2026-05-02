#include "DatasetLoader.h"
#include "FeatureMapper.h"
#include "Autoencoder.h"
#include "AnomalyDetector.h"
#include "FeatureStatistics.h"
#include <iostream>
#include <deque>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>


// helper functions

static void printAnomalyReport(const std::vector<double>& x, double anomalyScore, AnomalyDetector& ad, const FeatureStatistics& fs)
{
    // per-feature reconstruction errors
    auto allContribs = ad.explain(x, std::numeric_limits<int>::max());
    double totalError = 0.0;
    for (const auto& fc : allContribs) totalError += fc.error;

    const int REPORT_TOP_N = std::min(3, (int)allContribs.size());
    const double phi = ad.getPhi();

    // model confidence (how far above the threshold)
    double confidence = (phi > 1e-12) ? std::tanh((anomalyScore / phi - 1.0) * 2.0) * 100.0 : 0.0;

    // header
    std::cout << "\n==================== ANOMALY REPORT ====================\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Score:             " << anomalyScore << "\n";
    std::cout << "Anomaly threshold: " << phi << "\n";
    std::cout << "Decision:          ANOMALY\n\n";

    std::cout << std::setprecision(1);
    std::cout << "Model confidence:  " << confidence << "%\n";
    std::cout << "Note: Values are shown in normalized space; baselines are statistical estimates.\n\n";

    std::cout << "--------------------------------------------------------\n\n";

    
    std::cout << "TOP CONTRIBUTING FEATURES\n\n";

    // feature blocks
    std::vector<int> topFeatureIndices;
    std::vector<std::pair<int, double>> topFeatureZScores;

    for (int rank = 0; rank < REPORT_TOP_N; rank++) {
        const auto& fc = allContribs[rank];
        int fidx = fc.featureIndex;
        double val = x[fidx];
        double contribPct = (totalError > 1e-12) ? (fc.error / totalError * 100.0) : 0.0;

        const auto& st = fs.getStats(fidx);
        double z = fs.getZScore(fidx, val);

        // expected range
        double rangeLow = st.mean - 3.0 * st.stddev;
        double rangeHigh = st.mean + 3.0 * st.stddev;
        if (st.min >= 0.0) rangeLow = std::max(0.0, rangeLow);

        std::string featName = fs.getName(fidx);
        std::string reconLevel = fs.getReconstructionLevel(contribPct);

        std::cout << rank + 1 << ") Feature: " << featName << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Current value:            " << val << "\n";
        std::cout << "Baseline mean:            " << st.mean << "\n";
        std::cout << "Expected range:          [" << rangeLow << ", " << rangeHigh << "]\n";

        std::cout << std::setprecision(2);
        std::cout << "Z-score:                  " << z << "\n";

        std::cout << std::setprecision(1);
        std::cout << "Contribution (%):         " << contribPct << "%\n\n";

        std::cout << "Reconstruction contribution: " << reconLevel << "\n";

        std::cout << "Interpretation:\n";
        for (const auto& bullet : fs.getInterpretationBullets(fidx, val))
            std::cout << "- " << bullet << "\n";

        std::cout << "\nPossible causes:\n";
        for (const auto& bullet : fs.getPossibleCausesBullets(fidx))
            std::cout << "- " << bullet << "\n";

        std::cout << "\n--------------------------------------------------------\n\n";

        topFeatureIndices.push_back(fidx);
        topFeatureZScores.push_back({ fidx, z });
    }
    // finale
    std::cout << "SECURITY ASSESSMENT:\n\n";
    std::cout << "Risk level: " << fs.getRiskLevel(topFeatureZScores) << "\n\n";

    std::cout << "Most probable scenarios:\n";
    for (const auto& scenario : fs.getMostProbableScenarios(topFeatureIndices))
        std::cout << "- " << scenario << "\n";

    std::cout << "\nRECOMMENDED ACTIONS:\n\n";
    auto actions = fs.getRecommendedActions(topFeatureIndices);
    for (int i = 0; i < (int)actions.size(); i++)
        std::cout << i + 1 << ". " << actions[i] << "\n";

    std::cout << "\n========================================================\n\n";
    
}

// MAIN

int main() {
    DatasetLoader full("preprocessed_combined_dataset.csv");

    /* std::cout << "Features: " << full.getFeatureCount() << std::endl;
    for (int i = 0; i < full.getFeatureCount(); i++)
    {
        std::cout << full.getFeatureNames()[i] << std::endl;
    } */

    std::cout << "Interpreting data, please wait a few minutes..." << std::endl;

    int trainSize = 1000;
    std::vector<std::vector<double>> trainData;
    for (int i = 0; i < trainSize && full.hasNext(); i++) {
        trainData.push_back(full.next());
    }
    
    FeatureStatistics fs(full.getFeatureCount(), full.getFeatureNames());
    for (const auto& x : trainData) {
        fs.update(x);
    }
    fs.finalize();

    FeatureMapper fm(full.getFeatureCount(), 1); //2nd parameter is maxClusterSize
    fm.setFeatureNames(full.getFeatureNames());
    for (const auto& x : trainData) {
        fm.update(x);
    }
    fm.finalize();
    
    AnomalyDetector ad(fm.getClusters());
    for (const auto& x : trainData) {
        ad.train(x);
    }

    std::cout << "Anomaly threshold = " << ad.getPhi() << std::endl;

    std::deque<std::vector<double>> normalBuffer;
    const int BUFFER_SIZE = 1000; //buffer size for normal instances used for reclustering


    while (full.hasNext())
    {
        auto x = full.next();
        double anomalyScore = ad.score(x);
        bool anomaly = ad.isAnomaly(anomalyScore);

        if (anomaly)
        {
            printAnomalyReport(x, anomalyScore, ad, fs);
        }
        else {
            std::cout << "------------------------------------------------------\n";
            std::cout << "\n[Normal] Score: " << std::fixed << std::setprecision(4) << anomalyScore << "\n\n";
            std::cout << "------------------------------------------------------\n";
            normalBuffer.push_back(x);
            if (normalBuffer.size() > BUFFER_SIZE)
            {
                normalBuffer.pop_front();
            }
        }

        fm.updateIfNormal(x, !anomaly);

        if (fm.shouldRecluster())
        {
            std::cout << "Reclustering, please wait a few minutes..." << std::endl;
            fm.finalize();
            ad.rebuild(fm.getClusters());
            fm.resetStats();
            fm.sinceLastCluster = 0;
            for (auto& v : normalBuffer)
            {
                ad.train(v);
            }
            std::cout << "Anomaly threshold = " << ad.getPhi() << std::endl;
        }
    }
}
