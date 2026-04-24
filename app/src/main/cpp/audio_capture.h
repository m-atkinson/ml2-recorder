#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "timestamp.h"

namespace ml2 {

class VrsWriter;

struct AudioCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    int sample_rate = 48000;
    int channels    = 4;
    int bits_per_sample = 16;
};

/// Captures multi-channel audio from the ML2 microphones using OpenSL ES.
/// Delivers raw PCM chunks to VrsWriter as audio records.
class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    bool init(const AudioCaptureConfig& config);
    bool start();
    void stop();

    bool   is_running()    const { return running_.load(); }
    size_t bytes_written() const { return bytes_written_.load(); }

    // Called from OpenSL ES callback.
    void on_buffer(uint32_t buffer_index);

private:
    AudioCaptureConfig config_;

    std::atomic<bool>   running_{false};
    std::atomic<size_t> bytes_written_{0};

    void* sl_engine_obj_    = nullptr;
    void* sl_engine_        = nullptr;
    void* sl_recorder_obj_  = nullptr;
    void* sl_recorder_      = nullptr;
    void* sl_buffer_queue_  = nullptr;

    static constexpr int kNumBuffers       = 2;
    static constexpr int kBufferDurationMs = 20;
    std::vector<int16_t> buffers_[kNumBuffers];
    int buffer_size_samples_ = 0;
};

}  // namespace ml2
