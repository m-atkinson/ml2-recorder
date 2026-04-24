#pragma once

#include <string>
#include <vector>

namespace ml2 {

struct CaptureProfile;  // forward declaration

/// Writes metadata.json and calibration.json into the session directory.
/// Call write_metadata() at session start and finalize_metadata() at session
/// end to update end_time_ns and frame counts.
class MetadataWriter {
public:
    struct StreamInfo {
        std::string name;       // e.g., "rgb", "depth", "head_pose"
        bool enabled = false;
        size_t frame_count = 0; // filled by finalize
    };

    /// Per-camera intrinsic parameters.
    struct Intrinsics {
        double fx = 0.0, fy = 0.0;     // Focal length
        double cx = 0.0, cy = 0.0;     // Principal point
        double distortion[5] = {};      // Distortion coefficients
        bool valid = false;             // True if populated from hardware
    };

    /// Calibration data discovered at runtime from camera frame data.
    struct CalibrationData {
        int rgb_width = 0;
        int rgb_height = 0;
        int depth_width = 0;
        int depth_height = 0;
        int world_cam_width = 0;
        int world_cam_height = 0;

        Intrinsics rgb_intrinsics;
        Intrinsics depth_intrinsics;
        Intrinsics world_cam_intrinsics[3];  // left, right, center
    };

    /// Write the initial metadata.json with full profile settings.
    static bool write_metadata(
        const std::string& session_dir,
        int64_t start_time_ns,
        const std::vector<StreamInfo>& streams,
        const CaptureProfile& profile);

    /// Update metadata.json with end time and frame counts, preserving
    /// all fields from the initial write.
    static bool finalize_metadata(
        const std::string& session_dir,
        int64_t start_time_ns,
        int64_t end_time_ns,
        const std::vector<StreamInfo>& streams,
        const CaptureProfile& profile);

    /// Write calibration.json with actual dimensions from frame data.
    static bool write_calibration(const std::string& session_dir,
                                   const CalibrationData& cal);
};

} // namespace ml2
