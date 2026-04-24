#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "metadata_writer.h"
#include "timestamp.h"

typedef uint64_t MLHandle;

namespace ml2 {

class VrsWriter;

struct DepthCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    bool use_short_range  = false;
};

/// Captures depth and confidence frames from the ML2 ToF sensor.
/// Writes uint16 depth (mm) + optional uint16 confidence to VrsWriter.
/// The WriteQueue is no longer needed — float→uint16 conversion is done
/// inline in the capture loop and records are created directly.
class DepthCapture {
public:
    DepthCapture();
    ~DepthCapture();

    DepthCapture(const DepthCapture&) = delete;
    DepthCapture& operator=(const DepthCapture&) = delete;

    bool init(const DepthCaptureConfig& config);
    bool start();
    void stop();

    bool   is_running()     const { return running_.load(); }
    size_t frames_written() const { return frames_written_.load(); }
    size_t frames_dropped() const { return 0; }  // no queue, no drops

    uint32_t actual_width()  const { return actual_width_.load(); }
    uint32_t actual_height() const { return actual_height_.load(); }

    MetadataWriter::Intrinsics intrinsics() const { return intrinsics_; }

private:
    void capture_loop();

    DepthCaptureConfig config_;

    std::thread capture_thread_;
    std::atomic<bool>     running_{false};
    std::atomic<uint32_t> frame_index_{0};
    std::atomic<size_t>   frames_written_{0};
    std::atomic<uint32_t> actual_width_{0};
    std::atomic<uint32_t> actual_height_{0};

    MLHandle depth_handle_ = 0xFFFFFFFFFFFFFFFF;
    MetadataWriter::Intrinsics intrinsics_;
};

}  // namespace ml2
