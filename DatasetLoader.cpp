#include "DatasetLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

DatasetLoader::DatasetLoader(const std::string& filename, bool hasHeader) : filename(filename), hasHeader(hasHeader), index(0)
{
    loadFile();
}

void DatasetLoader::loadFile()
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("DatasetLoader: could not open file");
    }

    std::string line;
    bool firstRow = true;

    if (hasHeader) { //reading the header
        std::getline(file, line);
        featureNames = parseHeader(line);
        featureCount = 3130;
    }
    while (std::getline(file, line)) { //reading data rows
        if (line.empty()) continue;

        std::vector<double> row = parseLine(line);
        //std::cout << (int)row.size() << "\n";
        if (firstRow && !hasHeader) {
            featureCount = row.size();
            firstRow = false;
        }

        if ((int)row.size() != featureCount) {
            throw std::runtime_error("DatasetLoader: inconsistent column count in row");
        }

        data.push_back(row);
    }

    if (data.empty()) {
        throw std::runtime_error("DatasetLoader: dataset empty");
    }

    file.close();
}

std::vector<std::string> DatasetLoader::parseHeader(const std::string& line)
{
    std::vector<std::string> names;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        names.push_back(token);
    }

    if (names.empty()) {
        throw std::runtime_error("DatasetLoader: header row empty");
    }

    return names;
}

std::vector<double> DatasetLoader::parseLine(const std::string& line)
{
    std::vector<double> values;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {

        if (token == "True") { //converting booleans
            values.push_back(1.0);
        }
        else if (token == "False") {
            values.push_back(0.0);
        }
        else {
            try {
                values.push_back(std::stod(token));
            }
            catch (...) {
                throw std::runtime_error("DatasetLoader: invalid numeric value: " + token);
            }
        }
    }

    return values;
}


bool DatasetLoader::hasNext() const
{
    return index < data.size();
}

std::vector<double> DatasetLoader::next()
{
    if (!hasNext()) {
        throw std::out_of_range("DatasetLoader: out of range");
    }
    return data[index++];
}

int DatasetLoader::getFeatureCount() const
{
    return featureCount;
}

const std::vector<std::string>& DatasetLoader::getFeatureNames() const
{
    return featureNames;
}
