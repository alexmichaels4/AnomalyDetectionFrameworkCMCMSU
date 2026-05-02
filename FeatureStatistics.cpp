#include "FeatureStatistics.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <set>

// 1. construction

FeatureStatistics::FeatureStatistics(int featureCount, const std::vector<std::string>& names)
    : n(featureCount), names(names), sums(featureCount, 0.0), sumSqs(featureCount, 0.0), counts(featureCount, 0)
{
    stats.resize(featureCount);
}

void FeatureStatistics::update(const std::vector<double>& x) {
    for (int i = 0; i < n; ++i) {
        sums[i] += x[i];
        sumSqs[i] += x[i] * x[i];
        counts[i]++;
        if (x[i] < stats[i].min) stats[i].min = x[i];
        if (x[i] > stats[i].max) stats[i].max = x[i];
    }
}

void FeatureStatistics::finalize() {
    for (int i = 0; i < n; ++i) {
        if (counts[i] > 0) {
            stats[i].mean = sums[i] / counts[i];
            double var = (sumSqs[i] / counts[i]) - (stats[i].mean * stats[i].mean);
            stats[i].stddev = std::sqrt(var > 0 ? var : 0.0);
            if (stats[i].stddev < 1e-9) stats[i].stddev = 1e-9; // preventing division by zero
        }
    }
}

// 2. queries

bool FeatureStatistics::isSignificantDeviation(int idx, double value, double zThreshold) const {
    double z = std::abs(value - stats[idx].mean) / stats[idx].stddev;
    return z > zThreshold;
}

double FeatureStatistics::getZScore(int idx, double value) const {
    return (value - stats[idx].mean) / stats[idx].stddev;
}

std::string FeatureStatistics::getName(int idx) const {
    if (idx >= 0 && idx < (int)names.size()) return names[idx];
    return "Feature_" + std::to_string(idx);
}

const FeatureStats& FeatureStatistics::getStats(int idx) const {
    return stats[idx];
}

// 3. helper functions

std::string FeatureStatistics::toLower(const std::string& s) const {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return out;
}

FeatureStatistics::FeatureCategory FeatureStatistics::getCategory(int idx) const {
    std::string low = toLower(idx < (int)names.size() ? names[idx] : "");

    if (low.find("login") != std::string::npos) return FeatureCategory::LOGIN;
    if (low.find("response time") != std::string::npos ||
        low.find("duration") != std::string::npos) return FeatureCategory::RESPONSE_TIME;
    if (low.find("created files") != std::string::npos ||
        low.find("unique processes") != std::string::npos) return FeatureCategory::FILES_PROCESSES;
    if (low.find("packet") != std::string::npos ||
        low.find("external ip") != std::string::npos) return FeatureCategory::TRAFFIC_PACKETS;
    if (low.find("dns") != std::string::npos) return FeatureCategory::DNS;
    if (low.find("port") != std::string::npos) return FeatureCategory::PORT;
    if (low.find("peak") != std::string::npos ||
        low.find("lowest") != std::string::npos) return FeatureCategory::TIME_PATTERN;
    if (low.find("permission") != std::string::npos) return FeatureCategory::PERMISSION;
    if (low.find("log") != std::string::npos) return FeatureCategory::LOG;
    if (low.find("antivirus") != std::string::npos) return FeatureCategory::ANTIVIRUS;
    if (low.find("os") != std::string::npos ||
        low.find("kernel") != std::string::npos ||
        low.find("architecture") != std::string::npos ||
        low.find("ram") != std::string::npos) return FeatureCategory::SYSTEM_CONFIG;
    if (low.find("installed software") != std::string::npos) return FeatureCategory::SOFTWARE;
    if (low.find("termination") != std::string::npos) return FeatureCategory::TERMINATION;
    if (low.find("tls") != std::string::npos) return FeatureCategory::TLS;
    if (low.find("same ip") != std::string::npos) return FeatureCategory::SAME_IP;
    return FeatureCategory::UNKNOWN;
}

// 4. report building blocks

std::string FeatureStatistics::getReconstructionLevel(double pct) const {
    if (pct >= 40.0) return "CRITICAL";
    if (pct >= 20.0) return "HIGH";
    if (pct >= 10.0) return "MEDIUM";
    return "LOW";
}

std::vector<std::string>
FeatureStatistics::getInterpretationBullets(int idx, double value) const {
    auto cat = getCategory(idx);
    double z = getZScore(idx, value);
    std::string dir = (z > 0) ? "increase" : "decrease";

    switch (cat) {
    case FeatureCategory::LOGIN:
        return {
            "Extreme deviation in authentication frequency",
            "Behavior inconsistent with learned normal user patterns"
        };
    case FeatureCategory::RESPONSE_TIME:
        return {
            "Significant " + dir + " in response time compared to normal behavior",
            "Strong anomaly in temporal network performance profile"
        };
    case FeatureCategory::FILES_PROCESSES:
        return {
            "Notable " + dir + " in file creation or process activity",
            "Potential deviation in endpoint behavior patterns"
        };
    case FeatureCategory::TRAFFIC_PACKETS:
        return {
            "Abnormal traffic volume or connection frequency detected",
            "Potential data exfiltration or DDoS indicators observed"
        };
    case FeatureCategory::DNS:
        return {
            "Unusual DNS query pattern or frequency detected",
            "Possible covert channel or command-and-control communication"
        };
    case FeatureCategory::PORT:
        return {
            "Unexpected port exposure or scanning activity detected",
            "Possible unauthorized network reconnaissance or lateral movement"
        };
    case FeatureCategory::TIME_PATTERN:
        return {
            "Off-hours or atypical timing pattern detected",
            "Activity inconsistent with normal operational schedule"
        };
    case FeatureCategory::PERMISSION:
        return {
            "Unexpected change in access control settings detected",
            "Possible privilege escalation event"
        };
    case FeatureCategory::LOG:
        return {
            "Anomalous logging behavior detected",
            "Possible audit evasion or log tampering activity"
        };
    case FeatureCategory::ANTIVIRUS:
        return {
            "Unexpected endpoint antivirus state or configuration detected",
            "Possible interference with security services"
        };
    case FeatureCategory::SYSTEM_CONFIG:
        return {
            "Unusual system configuration or resource state detected",
            "Possible OS-level tampering or memory attack"
        };
    case FeatureCategory::SOFTWARE:
        return {
            "Unusual software composition on the endpoint detected",
            "Possible unauthorized tool installation or supply chain compromise"
        };
    case FeatureCategory::TERMINATION:
        return {
            "Abnormal session or protocol termination pattern detected",
            "Possible connection hijacking or firewall evasion technique"
        };
    case FeatureCategory::TLS:
        return {
            "Unexpected TLS version or cipher suite detected",
            "Possible protocol downgrade or misconfigured secure channel"
        };
    case FeatureCategory::SAME_IP:
        return {
            "Excessive requests from a single IP source detected",
            "Possible automated polling or single-source DoS behavior"
        };
    default:
        return {
            "Significant statistical deviation detected in feature value",
            "Further investigation required to assess impact"
        };
    }
}

std::vector<std::string> FeatureStatistics::getPossibleCausesBullets(int idx) const {
    auto cat = getCategory(idx);
    switch (cat) {
    case FeatureCategory::LOGIN:
        return {
            "Brute-force attack or credential stuffing",
            "Compromised account with automated access",
            "Unauthorized authentication automation (bot activity)"
        };
    case FeatureCategory::RESPONSE_TIME:
        return {
            "Network congestion or overload",
            "Denial-of-Service (DoS) or degraded service conditions",
            "Compromised endpoint causing processing delays"
        };
    case FeatureCategory::FILES_PROCESSES:
        return {
            "Malware or ransomware staging activity",
            "Automated script execution",
            "Unauthorized privilege escalation processes"
        };
    case FeatureCategory::TRAFFIC_PACKETS:
        return {
            "Distributed Denial-of-Service (DDoS) attack",
            "Data exfiltration or unauthorized external communication",
            "Botnet command-and-control (C2) traffic"
        };
    case FeatureCategory::DNS:
        return {
            "DNS tunneling or cache poisoning",
            "Automated C2 beaconing by malware",
            "Data exfiltration via DNS protocol"
        };
    case FeatureCategory::PORT:
        return {
            "Unauthorized port scanning or service discovery",
            "Lateral movement or backdoor deployment on non-standard ports",
            "Network reconnaissance activity"
        };
    case FeatureCategory::TIME_PATTERN:
        return {
            "Automated cron jobs or scheduled malicious scripts",
            "Off-hours data processing by unauthorized processes",
            "Resource-intensive malicious activity outside working hours"
        };
    case FeatureCategory::PERMISSION:
        return {
            "Privilege escalation attempt",
            "Unauthorized modification of sensitive resource permissions",
            "Misconfigured role-based access control (RBAC)"
        };
    case FeatureCategory::LOG:
        return {
            "Log tampering or deletion by a malicious actor",
            "Audit evasion technique",
            "Unauthorized SIEM configuration modification"
        };
    case FeatureCategory::ANTIVIRUS:
        return {
            "Intentional antivirus disablement by malware",
            "Corrupted or outdated antivirus signature database",
            "Malware interference with endpoint protection services"
        };
    case FeatureCategory::SYSTEM_CONFIG:
        return {
            "OS-level tampering or rootkit activity",
            "Unexpected virtualization or containerization change",
            "Memory exhaustion attack targeting core infrastructure"
        };
    case FeatureCategory::SOFTWARE:
        return {
            "Unauthorized tool installation by a threat actor",
            "Developer environment misuse",
            "Supply chain compromise via malicious packages"
        };
    case FeatureCategory::TERMINATION:
        return {
            "Connection hijacking attempt",
            "SYN flood or RST injection evasion technique",
            "Firewall rule bypass via protocol manipulation"
        };
    case FeatureCategory::TLS:
        return {
            "Protocol downgrade attack (e.g. POODLE / BEAST)",
            "Outdated or weak cipher suite in use",
            "Misconfigured secure communication channel"
        };
    case FeatureCategory::SAME_IP:
        return {
            "Automated polling or misconfigured retry logic",
            "Application-layer DoS from a single source",
            "Unauthorized high-frequency API or service access"
        };
    default:
        return {
            "Unexpected system or network behavior",
            "Potential security policy violation",
            "Additional forensic investigation required"
        };
    }
}

std::string FeatureStatistics::getRiskLevel(const std::vector<std::pair<int, double>>& featureZScores) const
{
    double maxAbsZ = 0.0;
    for (const auto& p : featureZScores)
    {
        maxAbsZ = std::max(maxAbsZ, std::abs(p.second));
    }
    if (maxAbsZ >= 8.0) return "CRITICAL";
    if (maxAbsZ >= 4.0) return "HIGH";
    if (maxAbsZ >= 2.0) return "MEDIUM";
    return "LOW";
}

std::vector<std::string> FeatureStatistics::getMostProbableScenarios(const std::vector<int>& featureIndices) const
{
    std::vector<std::string> scenarios;
    std::set<FeatureCategory> seen;

    for (int idx : featureIndices) {
        auto cat = getCategory(idx);
        if (seen.count(cat)) continue;
        seen.insert(cat);

        switch (cat) {
        case FeatureCategory::LOGIN:
            scenarios.push_back("Credential-based attack (credential stuffing / brute-force)");
            break;
        case FeatureCategory::RESPONSE_TIME:
            scenarios.push_back("Network stress or partial service degradation");
            break;
        case FeatureCategory::FILES_PROCESSES:
            scenarios.push_back("Potential malware or ransomware staging activity");
            break;
        case FeatureCategory::TRAFFIC_PACKETS:
            scenarios.push_back("Distributed Denial-of-Service or data exfiltration attempt");
            break;
        case FeatureCategory::DNS:
            scenarios.push_back("DNS-based covert channel or C2 beaconing");
            break;
        case FeatureCategory::PORT:
            scenarios.push_back("Unauthorized network scanning or lateral movement");
            break;
        case FeatureCategory::TIME_PATTERN:
            scenarios.push_back("Off-hours automated malicious script execution");
            break;
        case FeatureCategory::PERMISSION:
            scenarios.push_back("Privilege escalation or unauthorized access control change");
            break;
        case FeatureCategory::LOG:
            scenarios.push_back("Audit evasion or log tampering by a threat actor");
            break;
        case FeatureCategory::ANTIVIRUS:
            scenarios.push_back("Endpoint protection interference or malware-induced AV disablement");
            break;
        case FeatureCategory::SYSTEM_CONFIG:
            scenarios.push_back("OS-level tampering or system resource attack");
            break;
        case FeatureCategory::SOFTWARE:
            scenarios.push_back("Unauthorized software installation or supply chain compromise");
            break;
        case FeatureCategory::TERMINATION:
            scenarios.push_back("Session hijacking or firewall bypass attempt");
            break;
        case FeatureCategory::TLS:
            scenarios.push_back("TLS protocol downgrade or misconfigured encryption");
            break;
        case FeatureCategory::SAME_IP:
            scenarios.push_back("Single-source application-layer DoS or automated abuse");
            break;
        default:
            scenarios.push_back("Unclassified anomalous activity requiring forensic investigation");
            break;
        }
    }

    return scenarios;
}

std::vector<std::string> FeatureStatistics::getRecommendedActions(const std::vector<int>& featureIndices) const
{
    std::vector<std::string> actions;
    std::set<FeatureCategory> seen;
    std::set<std::string> added;

    auto add = [&](const std::string& a) {if (!added.count(a)) { added.insert(a); actions.push_back(a); }};

    for (int idx : featureIndices) {
        auto cat = getCategory(idx);
        if (seen.count(cat)) continue;
        seen.insert(cat);

        switch (cat) {
        case FeatureCategory::LOGIN:
            add("Immediately investigate authentication logs");
            add("Check for unusual login sources and IP addresses");
            break;
        case FeatureCategory::RESPONSE_TIME:
            add("Analyze network latency and traffic load patterns");
            add("Verify endpoint integrity and running services");
            break;
        case FeatureCategory::FILES_PROCESSES:
            add("Inspect processes generating excessive file operations");
            add("Scan endpoint for malware or unauthorized scripts");
            break;
        case FeatureCategory::TRAFFIC_PACKETS:
            add("Analyze network traffic patterns and firewall logs");
            add("Block or rate-limit suspicious traffic sources");
            break;
        case FeatureCategory::DNS:
            add("Inspect DNS query logs for tunneling or beaconing patterns");
            add("Apply DNS filtering and block suspicious domains");
            break;
        case FeatureCategory::PORT:
            add("Audit open ports and active network services");
            add("Investigate potential lateral movement paths");
            break;
        case FeatureCategory::TIME_PATTERN:
            add("Review scheduled tasks, cron jobs, and automation scripts");
            break;
        case FeatureCategory::PERMISSION:
            add("Audit recent permission changes and access control logs");
            break;
        case FeatureCategory::LOG:
            add("Verify log integrity and check for tampering or deletion");
            break;
        case FeatureCategory::ANTIVIRUS:
            add("Verify endpoint protection software status and update signatures");
            break;
        case FeatureCategory::SYSTEM_CONFIG:
            add("Check system configuration integrity and kernel-level activity");
            break;
        case FeatureCategory::SOFTWARE:
            add("Audit installed software inventory for unauthorized packages");
            break;
        case FeatureCategory::TERMINATION:
            add("Inspect connection termination patterns in network logs");
            break;
        case FeatureCategory::TLS:
            add("Audit TLS configurations and enforce strong cipher suites");
            break;
        case FeatureCategory::SAME_IP:
            add("Investigate high-frequency requests from single IP source");
            add("Apply rate limiting or temporary IP block if abuse is confirmed");
            break;
        default:
            add("Conduct full forensic investigation of the anomalous event");
            break;
        }
    }

    add("Escalate to security team for human review if automated response is insufficient");
    return actions;
}
