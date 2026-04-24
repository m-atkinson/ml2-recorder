#pragma once

#include <string>

namespace ml2 {

/// Capture profile: controls which sensors are active, target rates, and
/// encoding settings.  Use the static factory methods to create presets,
/// or build a custom profile programmatically.
struct CaptureProfile {
    std::string name;

    // --- Sensor enable flags ---
    bool rgb = true;
    bool depth = true;
    bool world_cams = true;
    bool head_pose = true;
    bool eye_tracking = true;
    bool hand_tracking = true;
    bool imu = true;
    bool audio = true;
    bool meshing = true;

    // --- RGB settings ---
    int rgb_width = 1440;
    int rgb_height = 1080;
    int rgb_fps = 15;
    int rgb_jpeg_quality = 90;
    bool rgb_use_h264 = true;       // Use hardware H.264 encoding (vs per-frame JPEG)
    int rgb_h264_bitrate = 8000000; // 8 Mbps for 1440x1080
    int rgb_writer_threads = 5;
    int rgb_queue_depth = 120;

    // --- World camera settings ---
    bool world_cam_left = true;
    bool world_cam_right = true;
    bool world_cam_center = true;
    bool world_cam_jpeg = true;    // JPEG for speed (PNG too slow for target rates)
    int world_cam_jpeg_quality = 85;
    int world_cam_writer_threads = 4;
    int world_cam_queue_depth = 90;

    // --- Perception settings ---
    int perception_poll_rate_hz = 30;

    // --- Depth settings ---
    bool depth_short_range = false;
    int depth_writer_threads = 2;
    int depth_queue_depth = 20;

    // --- IMU settings ---
    int imu_sample_period_us = 5000;  // 200 Hz

    // --- Audio settings ---
    int audio_channels = 4;
    int audio_sample_rate = 48000;

    // --- Meshing settings ---
    float meshing_poll_interval_s = 1.0f;   // Poll for mesh updates every N seconds
    float meshing_bounds_m = 5.0f;          // Half-extent of query AABB around head (meters)
    bool meshing_normals = true;            // Include vertex normals
    bool meshing_confidence = false;        // Include per-vertex confidence
    int meshing_lod = 1;                    // 0=min, 1=medium, 2=max detail

    // ─────────────────────────────────────────────────────────────────────
    // Factory presets
    // ─────────────────────────────────────────────────────────────────────

    /// Full quality: all sensors, maximum fidelity, PNG world cameras.
    static CaptureProfile full_quality() {
        CaptureProfile p;
        p.name = "full_quality";
        // All defaults are already "full quality".
        return p;
    }

    /// High temporal: all sensors, higher frame rates, JPEG world cameras
    /// for faster encoding, more writer threads.
    static CaptureProfile high_temporal() {
        CaptureProfile p;
        p.name = "high_temporal";
        p.rgb_fps = 30;
        p.rgb_writer_threads = 4;
        p.rgb_queue_depth = 90;
        p.world_cam_jpeg = true;
        p.world_cam_writer_threads = 5;
        p.world_cam_queue_depth = 120;
        p.perception_poll_rate_hz = 60;
        p.depth_writer_threads = 2;
        p.depth_queue_depth = 20;
        return p;
    }

    /// Lightweight: reduced sensor set for lower CPU/IO load.
    /// Center world cam only, depth off, lower perception rate.
    static CaptureProfile lightweight() {
        CaptureProfile p;
        p.name = "lightweight";
        p.depth = false;
        p.meshing = false;
        p.world_cam_left = false;
        p.world_cam_right = false;
        p.world_cam_center = true;
        p.world_cam_jpeg = true;
        p.world_cam_writer_threads = 1;
        p.world_cam_queue_depth = 10;
        p.perception_poll_rate_hz = 15;
        p.rgb_writer_threads = 2;
        p.imu_sample_period_us = 10000;  // 100 Hz
        return p;
    }

    /// Look up a profile by name. Returns full_quality() for unknown names.
    static CaptureProfile from_name(const std::string& name) {
        if (name == "high_temporal") return high_temporal();
        if (name == "lightweight")   return lightweight();
        return full_quality();  // default
    }
};

} // namespace ml2
