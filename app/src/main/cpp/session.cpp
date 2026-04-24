#include "session.h"

#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>

namespace ml2 {

static bool mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        std::string sub = path.substr(0, pos);
        mkdir(sub.c_str(), 0755);
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

bool Session::create(const std::string& base_dir) {
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);

    char buf[64];
    strftime(buf, sizeof(buf), "session_%Y%m%d_%H%M%S", &tm_buf);
    name_ = buf;
    path_ = base_dir + "/" + name_;

    return mkdir_p(path_);
}

std::string Session::create_subdir(const std::string& name) {
    std::string subdir = path_ + "/" + name;
    mkdir_p(subdir);
    return subdir;
}

std::string Session::filepath(const std::string& filename) const {
    return path_ + "/" + filename;
}

} // namespace ml2
