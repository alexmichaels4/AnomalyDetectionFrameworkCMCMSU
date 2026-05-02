#pragma once
#include <vector>
#include <string>

class DatasetLoader {
public:
    DatasetLoader(const std::string& filename, bool hasHeader = true);

    bool hasNext() const;
    std::vector<double> next();

    int getFeatureCount() const;
    const std::vector<std::string>& getFeatureNames() const;

private:
    std::string filename;
    bool hasHeader;
    int featureCount;
    size_t index;

    std::vector<std::vector<double>> data;
    std::vector<std::string> featureNames;

    void loadFile();
    std::vector<double> parseLine(const std::string& line);
    std::vector<std::string> parseHeader(const std::string& line);
};
