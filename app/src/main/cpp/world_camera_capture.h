#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "metadata_writer.h"
#include "timestamp.h"
#include "write_queue.h"

typedef uint64_t MLHandle;

namespace ml2 {

class VrsWriter;

struct WorldCameraCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    bool enable_left   = true;
    bool enable_right  = true;
    bool enable_center = true;
    int  jpeg_quality  = 85;
    // Each camera gets its own WriteQueue to prevent one camera's burst from
    // stalling the others (fixes the ~180 drops / 30 s Bug #2).
    int writer_threads = 2;
    int queue_depth    = 30;
};

/// Captures grayscale frames from the ML2's three world cameras.
/// JPEG-encodes each frame in a per-camera WriteQueue and writes to VrsWriter.
/// Three independent WriteQueues eliminate the cross-camera backpressure that
/// caused ~180 frame drops per 30-second session in the previous design.
class WorldCameraCapture {
public:
    WorldCameraCapture();
    ~WorldCameraCapture();

    WorldCameraCapture(const WorldCameraCapture&) = delete;
    WorldCameraCapture& operator=(const WorldCameraCapture&) = delete;

    bool init(const WorldCameraCaptureConfig& config);
    bool start();
    void stop();

    bool   is_running()              const { return running_.load(); }
    size_t frames_written(int cam)   const;
    size_t frames_dropped()          const;

    uint32_t actual_width()  const { return actual_width_.load(); }
    uint32_t actual_height() const { return actual_height_.load(); }

    MetadataWriter::Intrinsics intrinsics(int cam_index) const {
        if (cam_index < 0 || cam_index >= 3) return {};
        return intrinsics_[cam_index];
    }

private:
    void capture_loop();
    int  camera_id_to_index(uint32_t cam_id) const;

    static bool encode_jpeg(const uint8_t* data, uint32_t width, uint32_t height,
                             uint32_t stride, int quality,
                             std::vector<uint8_t>& out);

    WorldCameraCaptureConfig config_;

    // One WriteQueue per camera — each queue's encode fn writes to vrs_writer.
    std::unique_ptr<WriteQueue> write_queue_[3];
    std::atomic<size_t>   frames_written_[3] = {{0}, {0}, {0}};
    std::atomic<uint32_t> frame_index_[3]    = {{0}, {0}, {0}};

    // Cached stats for after queues are destroyed.
    size_t final_dropped_[3] = {0, 0, 0};

    std::thread capture_thread_;
    std::atomic<bool> running_{false};

    std::atomic<uint32_t> actual_width_{0};
    std::atomic<uint32_t> actual_height_{0};

    MLHandle world_cam_handle_ = 0xFFFFFFFFFFFFFFFF;
    MetadataWriter::Intrinsics intrinsics_[3];
};

}  // namespace ml2
