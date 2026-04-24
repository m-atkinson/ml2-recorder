#pragma once

#include <cstdint>
#include <string>

#include <jni.h>

#include "csv_writer.h"
#include "timestamp.h"

namespace ml2 {

/// Monitors device thermal state via Android PowerManager JNI.
/// Logs thermal data to thermal.csv and exposes a critical threshold check.
class ThermalMonitor {
public:
    /// Initialize and open thermal.csv in the session directory.
    /// Requires the JavaVM pointer and the Activity jobject for JNI access.
    bool init(const std::string& session_dir, JavaVM* vm, jobject activity);

    /// Sample the current temperature and write a row to thermal.csv.
    /// Returns the thermal status (0=none, 1=light, 2=moderate, 3=severe,
    /// 4=critical, 5=emergency, 6=shutdown).
    int sample();

    /// Returns true if thermal status >= threshold (default: SEVERE=3).
    bool is_critical(int threshold = 3);

    /// Close the CSV file.
    void close();

    /// Last sampled thermal status.
    int last_status() const { return last_status_; }

private:
    CsvWriter csv_;
    JavaVM* vm_ = nullptr;
    jobject activity_ = nullptr;
    int last_status_ = 0;
};

} // namespace ml2
