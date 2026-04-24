#include "metadata_writer.h"
#include "capture_profile.h"

#include <android/log.h>

#include <cstdio>
#include <cstring>
#include <sstream>

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

// ---------------------------------------------------------------------------
// Simple JSON helper — we avoid pulling in a JSON library for this.
// ---------------------------------------------------------------------------
static bool write_json_file(const std::string& path, const std::string& json) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        LOGE("Failed to write %s", path.c_str());
        return false;
    }
    std::fwrite(json.data(), 1, json.size(), f);
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Helper: write the full metadata JSON with all fields.
// ---------------------------------------------------------------------------
static std::string build_metadata_json(
        int64_t start_time_ns,
        int64_t end_time_ns,
        const std::vector<MetadataWriter::StreamInfo>& streams,
        const CaptureProfile& profile) {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"device\": \"Magic Leap 2\",\n";
    ss << "  \"capture_profile\": \"" << profile.name << "\",\n";
    ss << "  \"start_time_ns\": " << start_time_ns << ",\n";
    ss << "  \"end_time_ns\": " << end_time_ns << ",\n";

    // Enabled streams list.
    ss << "  \"enabled_streams\": [";
    bool first = true;
    for (const auto& s : streams) {
        if (s.enabled) {
            if (!first) ss << ", ";
            ss << "\"" << s.name << "\"";
            first = false;
        }
    }
    ss << "],\n";

    // Per-stream frame counts.
    ss << "  \"stream_frame_counts\": {\n";
    first = true;
    for (const auto& s : streams) {
        if (s.enabled) {
            if (!first) ss << ",\n";
            ss << "    \"" << s.name << "\": " << s.frame_count;
            first = false;
        }
    }
    ss << "\n  },\n";

    // Sensor settings from the capture profile.
    ss << "  \"sensor_settings\": {\n";
    ss << "    \"rgb\": {\n";
    ss << "      \"width\": " << profile.rgb_width << ",\n";
    ss << "      \"height\": " << profile.rgb_height << ",\n";
    ss << "      \"fps\": " << profile.rgb_fps << ",\n";
    ss << "      \"jpeg_quality\": " << profile.rgb_jpeg_quality << ",\n";
    ss << "      \"writer_threads\": " << profile.rgb_writer_threads << ",\n";
    ss << "      \"queue_depth\": " << profile.rgb_queue_depth << "\n";
    ss << "    },\n";

    ss << "    \"depth\": {\n";
    ss << "      \"short_range\": " << (profile.depth_short_range ? "true" : "false") << ",\n";
    ss << "      \"writer_threads\": " << profile.depth_writer_threads << ",\n";
    ss << "      \"queue_depth\": " << profile.depth_queue_depth << "\n";
    ss << "    },\n";

    ss << "    \"world_cameras\": {\n";
    ss << "      \"left\": " << (profile.world_cam_left ? "true" : "false") << ",\n";
    ss << "      \"right\": " << (profile.world_cam_right ? "true" : "false") << ",\n";
    ss << "      \"center\": " << (profile.world_cam_center ? "true" : "false") << ",\n";
    ss << "      \"encoding\": \"" << (profile.world_cam_jpeg ? "jpeg" : "png") << "\",\n";
    ss << "      \"jpeg_quality\": " << profile.world_cam_jpeg_quality << ",\n";
    ss << "      \"writer_threads\": " << profile.world_cam_writer_threads << ",\n";
    ss << "      \"queue_depth\": " << profile.world_cam_queue_depth << "\n";
    ss << "    },\n";

    ss << "    \"perception\": {\n";
    ss << "      \"poll_rate_hz\": " << profile.perception_poll_rate_hz << "\n";
    ss << "    },\n";

    ss << "    \"imu\": {\n";
    ss << "      \"sample_period_us\": " << profile.imu_sample_period_us << ",\n";
    ss << "      \"nominal_rate_hz\": " << (1000000 / profile.imu_sample_period_us) << "\n";
    ss << "    },\n";

    ss << "    \"audio\": {\n";
    ss << "      \"channels\": " << profile.audio_channels << ",\n";
    ss << "      \"sample_rate\": " << profile.audio_sample_rate << "\n";
    ss << "    }\n";

    ss << "  }\n";
    ss << "}\n";

    return ss.str();
}

// ---------------------------------------------------------------------------
// write_metadata
// ---------------------------------------------------------------------------

bool MetadataWriter::write_metadata(
        const std::string& session_dir,
        int64_t start_time_ns,
        const std::vector<StreamInfo>& streams,
        const CaptureProfile& profile) {
    return write_json_file(session_dir + "/metadata.json",
                            build_metadata_json(start_time_ns, 0, streams, profile));
}

// ---------------------------------------------------------------------------
// finalize_metadata — rewrite with actual end time and frame counts,
// preserving all fields from the initial write.
// ---------------------------------------------------------------------------

bool MetadataWriter::finalize_metadata(
        const std::string& session_dir,
        int64_t start_time_ns,
        int64_t end_time_ns,
        const std::vector<StreamInfo>& streams,
        const CaptureProfile& profile) {
    return write_json_file(session_dir + "/metadata.json",
                            build_metadata_json(start_time_ns, end_time_ns,
                                                streams, profile));
}

// ---------------------------------------------------------------------------
// write_calibration — populate with actual dimensions from frame data
// ---------------------------------------------------------------------------

// Helper to format a double with fixed precision.
static std::string fmtd(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.8f", v);
    return buf;
}

// Helper to write an intrinsics JSON block.
static void write_intrinsics_json(std::ostringstream& ss,
                                   const MetadataWriter::Intrinsics& intr,
                                   const char* indent) {
    ss << indent << "\"fx\": " << fmtd(intr.fx) << ",\n";
    ss << indent << "\"fy\": " << fmtd(intr.fy) << ",\n";
    ss << indent << "\"cx\": " << fmtd(intr.cx) << ",\n";
    ss << indent << "\"cy\": " << fmtd(intr.cy) << ",\n";
    ss << indent << "\"distortion\": ["
       << fmtd(intr.distortion[0]) << ", "
       << fmtd(intr.distortion[1]) << ", "
       << fmtd(intr.distortion[2]) << ", "
       << fmtd(intr.distortion[3]) << ", "
       << fmtd(intr.distortion[4]) << "]\n";
}

bool MetadataWriter::write_calibration(const std::string& session_dir,
                                        const CalibrationData& cal) {
    // Use provided dimensions, falling back to config defaults if zero.
    int rgb_w = cal.rgb_width > 0 ? cal.rgb_width : 1440;
    int rgb_h = cal.rgb_height > 0 ? cal.rgb_height : 1080;
    int depth_w = cal.depth_width;
    int depth_h = cal.depth_height;
    int wc_w = cal.world_cam_width;
    int wc_h = cal.world_cam_height;

    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"rgb\": {\n";
    ss << "    \"model\": \"pinhole\",\n";
    ss << "    \"width\": " << rgb_w << ",\n";
    ss << "    \"height\": " << rgb_h << ",\n";
    write_intrinsics_json(ss, cal.rgb_intrinsics, "    ");
    ss << "  },\n";
    ss << "  \"depth\": {\n";
    ss << "    \"model\": \"pinhole\",\n";
    ss << "    \"width\": " << depth_w << ",\n";
    ss << "    \"height\": " << depth_h << ",\n";
    write_intrinsics_json(ss, cal.depth_intrinsics, "    ");
    ss << "  },\n";

    const char* wc_ids[3] = {"left", "right", "center"};
    ss << "  \"world_cameras\": [\n";
    for (int i = 0; i < 3; ++i) {
        ss << "    {\n";
        ss << "      \"id\": \"" << wc_ids[i] << "\",\n";
        ss << "      \"model\": \"fisheye\",\n";
        ss << "      \"width\": " << wc_w << ",\n";
        ss << "      \"height\": " << wc_h << ",\n";
        write_intrinsics_json(ss, cal.world_cam_intrinsics[i], "      ");
        ss << "    }" << (i < 2 ? "," : "") << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";

    return write_json_file(session_dir + "/calibration.json", ss.str());
}

} // namespace ml2
