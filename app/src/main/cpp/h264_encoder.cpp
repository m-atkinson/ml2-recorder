#include "h264_encoder.h"

#include <android/log.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaFormat.h>

#include <cstring>

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

H264Encoder::H264Encoder() = default;

H264Encoder::~H264Encoder() {
    if (ready_) stop();
    if (codec_) AMediaCodec_delete(codec_);
}

// =============================================================================
// init
// =============================================================================

bool H264Encoder::init(const Config& config) {
    config_ = config;

    codec_ = AMediaCodec_createEncoderByType("video/avc");
    if (!codec_) {
        LOGE("H264Encoder: failed to create video/avc encoder");
        return false;
    }

    AMediaFormat* format = AMediaFormat_new();
    AMediaFormat_setString(format, AMEDIAFORMAT_KEY_MIME,         "video/avc");
    AMediaFormat_setInt32 (format, AMEDIAFORMAT_KEY_WIDTH,        config_.width);
    AMediaFormat_setInt32 (format, AMEDIAFORMAT_KEY_HEIGHT,       config_.height);
    AMediaFormat_setInt32 (format, AMEDIAFORMAT_KEY_BIT_RATE,     config_.bitrate);
    AMediaFormat_setFloat (format, AMEDIAFORMAT_KEY_FRAME_RATE,   (float)config_.fps);
    AMediaFormat_setInt32 (format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL,
                           config_.iframe_interval);
    // COLOR_FormatYUV420SemiPlanar = 21 (NV12) — required by ML2 HW encoder.
    AMediaFormat_setInt32 (format, AMEDIAFORMAT_KEY_COLOR_FORMAT, 21);

    media_status_t status = AMediaCodec_configure(
        codec_, format, nullptr, nullptr, AMEDIACODEC_CONFIGURE_FLAG_ENCODE);
    AMediaFormat_delete(format);

    if (status != AMEDIA_OK) {
        LOGE("H264Encoder: configure failed: %d", (int)status);
        AMediaCodec_delete(codec_);
        codec_ = nullptr;
        return false;
    }

    status = AMediaCodec_start(codec_);
    if (status != AMEDIA_OK) {
        LOGE("H264Encoder: start failed: %d", (int)status);
        AMediaCodec_delete(codec_);
        codec_ = nullptr;
        return false;
    }

    ready_ = true;
    LOGI("H264Encoder: sync mode %dx%d @ %d fps %d kbps drain_timeout=%d µs",
         config_.width, config_.height, config_.fps, config_.bitrate / 1000,
         config_.drain_timeout_us);
    return true;
}

// =============================================================================
// feed — non-blocking, called from camera copy thread
// =============================================================================

bool H264Encoder::feed(const uint8_t* yuv_data, size_t size,
                        int64_t timestamp_us) {
    if (!ready_ || eos_sent_) return false;

    // Drain output first: frees encoder pipeline slots before we ask for input.
    drain_output(false);

    // Non-blocking input dequeue (timeout=0).  Drops frame rather than block.
    ssize_t index = AMediaCodec_dequeueInputBuffer(codec_, 0);
    if (index < 0) {
        frames_dropped_++;
        LOGI("H264Encoder: drop frame %u (no input slot)", frames_fed_ + 1);
        return false;
    }

    size_t buf_size = 0;
    uint8_t* buf = AMediaCodec_getInputBuffer(codec_, (size_t)index, &buf_size);
    if (!buf) {
        LOGE("H264Encoder: getInputBuffer(%zd) null", index);
        AMediaCodec_queueInputBuffer(codec_, (size_t)index, 0, 0, timestamp_us, 0);
        return false;
    }

    // Convert I420 → NV12 into the codec buffer.
    const int w      = config_.width;
    const int h      = config_.height;
    const int y_size = w * h;
    const int uv_w   = w / 2;
    const int uv_h   = h / 2;
    const size_t nv12_size = (size_t)y_size + (size_t)uv_w * uv_h * 2;

    if (buf_size < nv12_size) {
        LOGE("H264Encoder: buf too small (%zu < %zu)", buf_size, nv12_size);
        AMediaCodec_queueInputBuffer(codec_, (size_t)index, 0, 0, timestamp_us, 0);
        return false;
    }

    // Y plane — straight copy.
    std::memcpy(buf, yuv_data, y_size);

    // U/V planes — interleave I420 planar → NV12 semi-planar.
    const uint8_t* u_src = yuv_data + y_size;
    const uint8_t* v_src = u_src + uv_w * uv_h;
    uint8_t* uv_dst = buf + y_size;
    for (int i = 0; i < uv_w * uv_h; ++i) {
        uv_dst[i * 2]     = u_src[i];
        uv_dst[i * 2 + 1] = v_src[i];
    }

    AMediaCodec_queueInputBuffer(codec_, (size_t)index, 0, nv12_size,
                                  timestamp_us, 0);
    frames_fed_++;
    return true;
}

// =============================================================================
// stop
// =============================================================================

void H264Encoder::stop() {
    if (!ready_) return;
    LOGI("H264Encoder: stopping, %u frames fed", frames_fed_);

    // Send EOS on a real input buffer so the encoder flushes remaining frames.
    if (!eos_sent_) {
        // Give encoder up to 100 ms to free an input slot.
        ssize_t index = AMediaCodec_dequeueInputBuffer(codec_, 100000);
        if (index >= 0) {
            AMediaCodec_queueInputBuffer(codec_, (size_t)index, 0, 0, 0,
                                         AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
            eos_sent_ = true;
            LOGI("H264Encoder: EOS queued on slot %zd", index);
        } else {
            LOGE("H264Encoder: no input slot for EOS after 100 ms — flushing");
        }
    }

    // Drain remaining output (including EOS marker) for up to 2 s.
    drain_output(true);

    AMediaCodec_stop(codec_);
    ready_ = false;

    LOGI("H264Encoder: stopped — fed=%u written=%u dropped=%u",
         frames_fed_, frames_written_, frames_dropped_);
}

// =============================================================================
// drain_output — dequeue and deliver output buffers until none are ready
// =============================================================================

void H264Encoder::drain_output(bool end_of_stream) {
    const int64_t timeout_us = end_of_stream ? 500000 : config_.drain_timeout_us;

    while (true) {
        AMediaCodecBufferInfo info{};
        ssize_t index = AMediaCodec_dequeueOutputBuffer(codec_, &info, timeout_us);

        if (index == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            LOGI("H264Encoder: output format changed");
            continue;
        }
        if (index == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
            continue;
        }
        if (index < 0) {
            // No output available (or error).
            break;
        }

        if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
            LOGI("H264Encoder: EOS in output — %u frames written", frames_written_);
            AMediaCodec_releaseOutputBuffer(codec_, (size_t)index, false);
            break;
        }

        if (info.size > 0 && config_.nal_callback) {
            size_t buf_size = 0;
            uint8_t* buf = AMediaCodec_getOutputBuffer(codec_, (size_t)index,
                                                        &buf_size);
            if (buf) {
                const bool is_config =
                    (info.flags & AMEDIACODEC_BUFFER_FLAG_CODEC_CONFIG) != 0;
                const bool is_keyframe = (info.flags & 1) != 0;

                // ── ML2 keyframe buffer workaround ──────────────────────
                // On the ML2 hardware encoder, keyframe output buffers
                // have the layout:
                //   [0..26]  SPS + PPS (Annex B, with start codes)
                //   [27..]   Raw IDR slice data (NO start code, NO NAL
                //            header — the byte at info.offset has
                //            forbidden_zero_bit=1, making it invalid).
                //
                // info.offset is 27 for these keyframes, pointing past
                // the SPS/PPS. However, the data at that offset is not a
                // valid standalone NAL unit.
                //
                // Fix: for keyframe buffers with a non-zero offset, emit
                // the FULL buffer (from offset 0) as a single access unit.
                // This gives the decoder: SPS → PPS → raw IDR data, which
                // is a valid Annex B stream.  The separate CODEC_CONFIG
                // record (SPS/PPS) is still emitted independently.
                const uint8_t* nal_data;
                size_t nal_size;

                if (is_keyframe && info.offset > 0) {
                    // Use the full buffer including inline SPS/PPS.
                    nal_data = buf;
                    nal_size = (size_t)(info.offset + info.size);

                    LOGI("H264Encoder: keyframe — using full buffer "
                         "(offset=%d info.size=%d total=%zu "
                         "first8=%02x%02x%02x%02x%02x%02x%02x%02x)",
                         info.offset, info.size, nal_size,
                         nal_data[0], nal_data[1],
                         nal_size > 2 ? nal_data[2] : 0,
                         nal_size > 3 ? nal_data[3] : 0,
                         nal_size > 4 ? nal_data[4] : 0,
                         nal_size > 5 ? nal_data[5] : 0,
                         nal_size > 6 ? nal_data[6] : 0,
                         nal_size > 7 ? nal_data[7] : 0);
                } else {
                    nal_data = buf + info.offset;
                    nal_size = (size_t)info.size;
                }

                // ── Annex B start-code enforcement ──────────────────────
                // Ensure the output always starts with an Annex B start
                // code.  Some encoder implementations omit it.
                static constexpr uint8_t kStartCode[4] = {0, 0, 0, 1};
                const bool has_4byte_sc =
                    nal_size >= 4 &&
                    nal_data[0] == 0 && nal_data[1] == 0 &&
                    nal_data[2] == 0 && nal_data[3] == 1;
                const bool has_3byte_sc =
                    !has_4byte_sc && nal_size >= 3 &&
                    nal_data[0] == 0 && nal_data[1] == 0 && nal_data[2] == 1;

                std::vector<uint8_t> patched_buf;
                if (!has_4byte_sc && !has_3byte_sc) {
                    patched_buf.resize(4 + nal_size);
                    std::memcpy(patched_buf.data(), kStartCode, 4);
                    std::memcpy(patched_buf.data() + 4, nal_data, nal_size);
                    nal_data = patched_buf.data();
                    nal_size = patched_buf.size();
                    LOGI("H264Encoder: prepended start code to NAL "
                         "(config=%d key=%d orig_size=%zu)",
                         is_config, is_keyframe, (size_t)info.size);
                }

                // Diagnostic logging for keyframes/config NALs.
                if (is_config || is_keyframe || frames_written_ < 3) {
                    int sc_len = has_4byte_sc ? 4 : (has_3byte_sc ? 3 : 4);
                    uint8_t nal_header = (nal_size > (size_t)sc_len)
                                         ? nal_data[sc_len] : 0;
                    int nal_type = nal_header & 0x1F;
                    const char* type_name =
                        nal_type == 1 ? "P-slice" :
                        nal_type == 5 ? "IDR" :
                        nal_type == 6 ? "SEI" :
                        nal_type == 7 ? "SPS" :
                        nal_type == 8 ? "PPS" :
                        nal_type == 9 ? "AUD" : "other";
                    LOGI("H264Encoder: NAL %zu bytes type=%d(%s) "
                         "config=%d key=%d flags=0x%x offset=%d n=%u",
                         nal_size, nal_type, type_name,
                         is_config, is_keyframe, info.flags,
                         info.offset, frames_written_ + 1);
                }

                config_.nal_callback(nal_data, nal_size,
                                     is_config, is_keyframe,
                                     info.presentationTimeUs);
                if (!is_config) {
                    ++frames_written_;
                }
            }
        }

        AMediaCodec_releaseOutputBuffer(codec_, (size_t)index, false);

        // In non-EOS drain mode, stop after consuming one buffer so we don't
        // spend all frame budget here.  In EOS mode, keep going until done.
        if (!end_of_stream) break;
    }
}

}  // namespace ml2
