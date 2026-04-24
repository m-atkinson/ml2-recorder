#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// VRS types are deliberately NOT included here.  All VRS plumbing lives in
// vrs_writer.cpp behind the Pimpl.  Capture class headers only need to
// forward-declare VrsWriter.

namespace ml2 {

/// Thread-safe wrapper around vrs::RecordFileWriter.
///
/// One VrsWriter is created per session.  All capture classes hold a
/// VrsWriter* and call the appropriate write_* method from their sensor
/// threads.  VRS's internal createRecord() is thread-safe; the background
/// flush thread (owned by RecordFileWriter) serialises writes to disk.
///
/// Stream IDs / RecordableTypeIds are chosen once here and are constant for
/// the lifetime of the file.  pyvrs can enumerate them by name via the
/// per-stream flavor tags we write in the config records.
class VrsWriter {
public:
    VrsWriter();
    ~VrsWriter();

    VrsWriter(const VrsWriter&) = delete;
    VrsWriter& operator=(const VrsWriter&) = delete;

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Create the .vrs file at |filepath| and register all streams.
    /// Must be called before any write_* method.
    bool open(const std::string& filepath);

    /// Flush remaining records, write the file index, close the file.
    /// Sets file-level tags (metadata) before closing.
    void close();

    bool is_open() const;

    // -----------------------------------------------------------------------
    // File-level metadata (written as VRS file tags at close)
    // -----------------------------------------------------------------------

    void set_tag(const std::string& key, const std::string& value);

    // -----------------------------------------------------------------------
    // Per-stream configuration (write config records before data)
    // -----------------------------------------------------------------------

    struct RgbConfig {
        int width = 0, height = 0, fps = 0, bitrate = 0;
        // Intrinsics are captured from the first SDK frame, so the config
        // record written at session start leaves these zero. A second
        // config record with populated intrinsics is emitted at close.
        float fx = 0, fy = 0, cx = 0, cy = 0;
        float distortion[5] = {};
    };
    void write_rgb_config(const RgbConfig& cfg);

    struct DepthConfig {
        int width = 0, height = 0;
        float fx = 0, fy = 0, cx = 0, cy = 0;
        float distortion[5] = {};
    };
    void write_depth_config(const DepthConfig& cfg);

    struct WorldCamConfig {
        int cam_index = 0;  // 0=left, 1=right, 2=center
        int width = 0, height = 0;
        float fx = 0, fy = 0, cx = 0, cy = 0;
        float distortion[5] = {};
    };
    void write_world_cam_config(const WorldCamConfig& cfg);

    struct AudioConfig {
        int sample_rate = 48000;
        int channels = 4;
    };
    void write_audio_config(const AudioConfig& cfg);

    // -----------------------------------------------------------------------
    // Data writes — called from capture threads
    // -----------------------------------------------------------------------

    /// Per-frame camera extrinsic (world_from_camera).
    /// Quaternion order is SDK-native: x, y, z, w (scalar-last).
    /// Pass nullptr to write the record with camera_pose_valid=0.
    struct CameraPose {
        float position[3];     // metres
        float orientation[4];  // x, y, z, w
    };

    /// RGB: one H.264 NAL unit (Annex B, including codec-config / SPS+PPS).
    void write_rgb_nal(int64_t timestamp_ns, int64_t sensor_ts_ns,
                       uint32_t frame_index,
                       const uint8_t* nal_data, size_t nal_size,
                       bool is_config,   // true = SPS/PPS config unit
                       bool is_keyframe,
                       const CameraPose* pose = nullptr);

    /// Depth: raw float32 depth (metres) + optional float32 confidence.
    /// Internally converted to uint16 mm for compact storage.
    void write_depth_frame(int64_t timestamp_ns, int64_t sensor_ts_ns,
                           uint32_t width, uint32_t height,
                           uint32_t frame_index,
                           const float* depth_m,
                           const float* confidence,  // may be nullptr
                           const CameraPose* pose = nullptr);

    /// World camera: JPEG-encoded grayscale frame from cam_index {0,1,2}.
    void write_world_cam_frame(int cam_index,
                               int64_t timestamp_ns, int64_t sensor_ts_ns,
                               uint32_t width, uint32_t height,
                               uint32_t frame_index,
                               const uint8_t* jpeg_data, size_t jpeg_size,
                               const CameraPose* pose = nullptr);

    /// Head pose: position (m) + orientation quaternion (w,x,y,z).
    void write_head_pose(int64_t timestamp_ns, int64_t sensor_ts_ns,
                         float px, float py, float pz,
                         float qw, float qx, float qy, float qz);

    /// Eye tracking: left+right ray origins & directions, fixation point.
    void write_eye_tracking(int64_t timestamp_ns, int64_t sensor_ts_ns,
                            float left_ox, float left_oy, float left_oz,
                            float left_dx, float left_dy, float left_dz,
                            float right_ox, float right_oy, float right_oz,
                            float right_dx, float right_dy, float right_dz,
                            float fix_x, float fix_y, float fix_z);

    /// Hand tracking: 28 keypoints × 2 hands (3 floats each, NaN = invalid).
    /// left_kp / right_kp must each point to 28*3 floats.
    void write_hand_tracking(int64_t timestamp_ns, int64_t sensor_ts_ns,
                             bool left_valid, bool right_valid,
                             float left_confidence, float right_confidence,
                             const float* left_kp,   // 28*3 floats
                             const float* right_kp); // 28*3 floats

    /// IMU: one sample from one sensor unit.
    void write_imu(int64_t timestamp_ns, int64_t sensor_ts_ns,
                   int unit_id,
                   float ax, float ay, float az,
                   float gx, float gy, float gz);

    /// Audio: raw PCM samples (interleaved channels, int16).
    void write_audio(int64_t timestamp_ns,
                     const int16_t* samples, uint32_t num_samples,
                     uint32_t num_channels);

    /// Spatial mesh: raw mesh geometry (not PLY-encoded).
    /// vertices = float[vertex_count * 3], indices = uint32[index_count],
    /// normals = float[vertex_count * 3] (may be nullptr).
    void write_mesh(int64_t timestamp_ns,
                    uint32_t snapshot_index,
                    uint32_t vertex_count, uint32_t index_count,
                    const float* vertices,
                    const uint32_t* indices,
                    const float* normals);  // may be nullptr

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ml2
