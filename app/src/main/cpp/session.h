#pragma once

#include <cstdint>
#include <string>

namespace ml2 {

// Creates and manages the session output directory.
// Generates the session_YYYYMMDD_HHMMSS directory structure.
class Session {
public:
    // Base path where all sessions are stored (e.g., /sdcard/ml2_recordings)
    static constexpr const char* kBaseDir = "/sdcard/ml2_recordings";

    // Create a new session directory with a timestamped name.
    // base_dir defaults to kBaseDir; override for testing on host.
    bool create(const std::string& base_dir = kBaseDir);

    // Get the full path to the session directory.
    const std::string& path() const { return path_; }

    // Get the session name (e.g., "session_20260101_120000")
    const std::string& name() const { return name_; }

    // Create a subdirectory within the session (e.g., "rgb", "depth").
    // Returns the full path to the created subdirectory.
    std::string create_subdir(const std::string& name);

    // Get the path to a file within the session directory.
    std::string filepath(const std::string& filename) const;

    // Path to the single VRS output file for this session.
    // e.g., /sdcard/ml2_recordings/session_20260101_120000.vrs
    std::string vrs_path() const { return path_ + ".vrs"; }

private:
    std::string path_;
    std::string name_;
};

} // namespace ml2
