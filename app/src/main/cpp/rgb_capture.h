#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "h264_encoder.h"
#include "metadata_writer.h"
#include "timestamp.h"
#include "vrs_writer.h"
#include "write_queue.h"

typedef uint64_t MLHandle;
typedef uint64_t MLCameraContext;
struct MLCameraOutput;
struct MLCameraResultExtras;

namespace ml2 {

struct RgbCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    int  width          = 1440;
    int  height         = 1080;
    int  fps            = 15;
    int  jpeg_quality   = 90;
    bool use_h264       = true;
    int  h264_bitrate   = 8000000;
    int  writer_threads = 2;   // JPEG fallback path only
    int  queue_depth    = 30;  // JPEG fallback path only
};

/// Captures RGB frames from the ML2 CV camera.
///
/// H.264 path (default, two-thread pipeline):
///   camera callback → pending_frame_ → copy_thread_ (DMA copy into pre-alloc buffer)
///     → pending_encode_ → encode_thread_ (I420→NV12, H.264 feed, NAL→VRS)
///
/// Decoupling copy and encode means the DMA read (50–150 ms) no longer
/// serialises with the encoder feed (4 ms).  encode_stalls_ > 0 would indicate
/// the encoder is unexpectedly falling behind (should never happen in practice).
///
/// JPEG fallback: per-frame JPEG encode via WriteQueue, written to VrsWriter.
class RgbCapture {
public:
    RgbCapture();
    ~RgbCapture();

    RgbCapture(const RgbCapture&) = delete;
    RgbCapture& operator=(const RgbCapture&) = delete;

    bool init(const RgbCaptureConfig& config);
    bool start();
    void stop();

    /// Wire in the head-tracking handle owned by PerceptionCapture. Must be
    /// called before start() to enable per-frame extrinsic capture via
    /// MLCVCameraGetFramePose. Called with 0xFFFFFFFFFFFFFFFF or left unset,
    /// frames are written with camera_pose_valid=0.
    void set_head_handle(MLHandle handle) { head_handle_ = handle; }

    size_t frames_written() const;
    size_t frames_dropped() const;
    bool   is_running()     const { return running_.load(); }
    float  actual_fps()     const;

    /// Diagnostic counters for pose-lookup health.
    uint32_t pose_lookup_ok()   const { return pose_lookup_ok_.load(); }
    uint32_t pose_lookup_fail() const { return pose_lookup_fail_.load(); }

    MetadataWriter::Intrinsics intrinsics() const {
        std::lock_guard<std::mutex> lock(intrinsics_mutex_);
        return intrinsics_;
    }

private:
    static void on_video_buffer(const MLCameraOutput* output,
                                const MLHandle metadata_handle,
                                const MLCameraResultExtras* extra,
                                void* data);
    void copy_loop();
    void encode_loop();  // H.264 path only — feeds encoder off the DMA copy thread

    static bool write_jpeg(const uint8_t* rgba_data, int width, int height,
                           const std::string& filepath, int quality);
    static bool write_yuv_jpeg(const uint8_t* yuv_data, int width, int height,
                                const std::string& filepath, int quality);

    RgbCaptureConfig config_;
    std::unique_ptr<WriteQueue>   write_queue_;   // JPEG fallback
    std::unique_ptr<H264Encoder>  h264_encoder_;  // H.264 path

    std::atomic<bool>     running_{false};
    std::atomic<uint32_t> frame_index_{0};

    size_t final_written_ = 0;
    size_t final_dropped_ = 0;

    MLCameraContext camera_context_  = 0;
    bool            camera_connected_ = false;

    // CV camera tracking: queries per-frame camera_pose in the ML2 world frame.
    // head_handle_ is supplied externally (PerceptionCapture owns the tracker).
    MLHandle              cv_tracking_handle_ = 0xFFFFFFFFFFFFFFFF;
    MLHandle              head_handle_        = 0xFFFFFFFFFFFFFFFF;
    std::atomic<uint32_t> pose_lookup_ok_{0};
    std::atomic<uint32_t> pose_lookup_fail_{0};
    // Encoder emits NALs asynchronously. Map frame PTS (µs) → pose so the NAL
    // callback can attach the correct extrinsic. Pruned on insert.
    std::mutex                                            pose_map_mutex_;
    std::unordered_map<int64_t, VrsWriter::CameraPose>    pose_map_;

    std::mutex              camera_available_mutex_;
    std::condition_variable camera_available_cv_;
    bool                    camera_available_ = false;

    mutable std::mutex         intrinsics_mutex_;
    MetadataWriter::Intrinsics intrinsics_;

    std::atomic<int64_t>  first_frame_ns_{0};
    std::atomic<int64_t>  last_frame_ns_{0};
    std::atomic<uint32_t> callback_count_{0};
    std::atomic<int64_t>  first_callback_ns_{0};
    std::atomic<int64_t>  last_callback_ns_{0};
    std::atomic<int64_t>  max_copy_ns_{0};
    std::atomic<int64_t>  max_callback_ns_{0};

    // ── Camera → copy thread handoff ────────────────────────────────────────
    struct PendingFrame {
        const MLCameraOutput* output = nullptr;
        int64_t  common_ts   = 0;
        int64_t  sensor_ts   = 0;
        uint32_t frame_index = 0;
        bool     ready       = false;
        // World-from-camera, captured synchronously in on_video_buffer.
        uint8_t  pose_valid  = 0;
        float    pose_position[3]    = {0, 0, 0};
        float    pose_orientation[4] = {0, 0, 0, 0};  // x, y, z, w
    };
    std::mutex              copy_mutex_;
    std::condition_variable copy_cv_;
    PendingFrame            pending_frame_;
    std::thread             copy_thread_;
    std::atomic<bool>       copy_thread_running_{false};
    std::atomic<uint32_t>   copy_stalls_{0};

    // ── Copy thread → encoder thread handoff (H.264 path only) ──────────────
    // Two pre-allocated YUV buffers alternate: copy_thread_ writes into
    // encode_buf_[copy_buf_idx_] while encode_thread_ reads the other one.
    // Because encode (~4 ms) is much faster than DMA copy (~80 ms), the encoder
    // is always done before the copy thread needs to reuse its buffer.
    struct PendingEncode {
        uint8_t* data         = nullptr;  // points into encode_buf_[0] or [1]
        size_t   size         = 0;
        int64_t  timestamp_us = 0;
        uint32_t frame_index  = 0;
        bool     ready        = false;
    };
    std::vector<uint8_t>    encode_buf_[2];
    int                     copy_buf_idx_  = 0;
    std::mutex              encode_mutex_;
    std::condition_variable encode_cv_;
    PendingEncode           pending_encode_;
    std::thread             encode_thread_;
    std::atomic<bool>       encode_thread_running_{false};
    std::atomic<uint32_t>   encode_stalls_{0};  // should stay 0 in normal operation
};

}  // namespace ml2
