#pragma once

#include <cstdint>
#include <functional>
#include <string>

// Forward-declare NDK types to avoid including media headers everywhere.
struct AMediaCodec;

namespace ml2 {

/// Hardware H.264 encoder using Android NDK MediaCodec (synchronous mode).
///
/// Accepts raw YUV I420 frames and encodes them to H.264 Annex B NAL units,
/// delivering each NAL unit to the caller via NalCallback.
///
/// Call pattern per frame:
///   feed()  →  drain_output() is called before AND after queuing input.
///              feed() is non-blocking: if no input slot is available after
///              draining, the frame is dropped (never waits > drain_timeout_us).
///
/// Thread safety: NOT thread-safe.  All calls must be serialised by the caller
/// (rgb_capture copy_loop thread).
class H264Encoder {
public:
    using NalCallback = std::function<void(
        const uint8_t* data, size_t size,
        bool is_config, bool is_keyframe, int64_t pts_us)>;

    struct Config {
        int width  = 1440;
        int height = 1080;
        int fps    = 15;
        int bitrate       = 8000000;
        int iframe_interval = 1;
        // How long to wait for an output buffer when draining (µs).
        // A short positive value (2 ms) lets the HW encoder produce output
        // without blocking the copy thread for a full frame budget.
        int drain_timeout_us = 0;
        NalCallback nal_callback;
    };

    H264Encoder();
    ~H264Encoder();

    H264Encoder(const H264Encoder&) = delete;
    H264Encoder& operator=(const H264Encoder&) = delete;

    bool init(const Config& config);

    /// Feed one YUV I420 frame.  Returns false and drops frame if no input
    /// slot is available after draining; never blocks longer than
    /// drain_timeout_us regardless of encoder throughput.
    bool feed(const uint8_t* yuv_data, size_t size, int64_t timestamp_us);

    void stop();

    uint32_t frames_fed()     const { return frames_fed_; }
    uint32_t frames_written() const { return frames_written_; }
    uint32_t frames_dropped() const { return frames_dropped_; }
    bool     is_ready()       const { return ready_; }

private:
    void drain_output(bool end_of_stream);

    Config config_;
    AMediaCodec* codec_ = nullptr;
    bool ready_     = false;
    bool eos_sent_  = false;
    uint32_t frames_fed_     = 0;
    uint32_t frames_written_ = 0;
    uint32_t frames_dropped_ = 0;
};

}  // namespace ml2
