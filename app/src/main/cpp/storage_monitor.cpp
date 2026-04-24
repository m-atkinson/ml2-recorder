#include "storage_monitor.h"

#include <sys/statvfs.h>

namespace ml2 {

int64_t StorageMonitor::available_mb(const std::string& path) {
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) != 0) {
        return -1;
    }
    // Available blocks * block size, converted to MB.
    return static_cast<int64_t>(stat.f_bavail) * stat.f_frsize / (1024 * 1024);
}

bool StorageMonitor::is_low(const std::string& path, int64_t threshold_mb) {
    int64_t avail = available_mb(path);
    return avail >= 0 && avail < threshold_mb;
}

} // namespace ml2
