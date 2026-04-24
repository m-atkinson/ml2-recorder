#pragma once

#include <cstdint>
#include <string>

namespace ml2 {

/// Monitors available disk space on the recording path.
class StorageMonitor {
public:
    /// Returns available space in MB for the filesystem containing |path|.
    /// Returns -1 on error.
    static int64_t available_mb(const std::string& path);

    /// Returns true if available space is below |threshold_mb|.
    static bool is_low(const std::string& path, int64_t threshold_mb = 1024);
};

} // namespace ml2
