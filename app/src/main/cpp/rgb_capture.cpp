#include "rgb_capture.h"

#include <android/log.h>

#include <chrono>
#include <cstring>
#include <vector>

#include <ml_camera_v2.h>
#include <ml_cv_camera.h>
#include <ml_types.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "fast_copy.h"
#include "stb_image_write.h"

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

RgbCapture::RgbCapture() = default;
RgbCapture::~RgbCapture() { stop(); }

bool RgbCapture::init(const RgbCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("RgbCapture: vrs_writer is null");
        return false;
    }

    // Availability callbacks.
    MLCameraDeviceAvailabilityStatusCallbacks avail_cbs = {};
    MLCameraDeviceAvailabilityStatusCallbacksInit(&avail_cbs);

    avail_cbs.on_device_available =
        [](const MLCameraDeviceAvailabilityInfo* info) {
            if (!info) return;
            auto* self = static_cast<RgbCapture*>(info->user_data);
            if (info->cam_id == MLCameraIdentifier_CV ||
                info->cam_id == MLCameraIdentifier_MAIN) {
                std::lock_guard<std::mutex> lock(self->camera_available_mutex_);
                self->camera_available_ = true;
                self->camera_available_cv_.notify_one();
            }
        };
    avail_cbs.on_device_unavailable =
        [](const MLCameraDeviceAvailabilityInfo* info) {
            if (info) LOGE("Camera unavailable: cam_id=%d",
                           static_cast<int>(info->cam_id));
        };

    MLResult result = MLCameraInit(&avail_cbs, this);
    if (result != MLResult_Ok) {
        LOGE("MLCameraInit failed: %d", static_cast<int>(result));
        return false;
    }

    // Wait for camera availability (5 s).
    {
        std::unique_lock<std::mutex> lock(camera_available_mutex_);
        if (!camera_available_cv_.wait_for(lock, std::chrono::seconds(5),
                [this] { return camera_available_; })) {
            LOGE("Timeout waiting for CV camera");
            MLCameraDeInit();
            return false;
        }
    }

    // Connect.
    MLCameraConnectContext ctx = {};
    MLCameraConnectContextInit(&ctx);
    ctx.cam_id = MLCameraIdentifier_CV;
    ctx.flags  = MLCameraConnectFlag_CamOnly;
    ctx.enable_video_stab = false;

    result = MLCameraConnect(&ctx, &camera_context_);
    if (result != MLResult_Ok) {
        LOGI("CV camera connect failed (%d), trying MAIN", static_cast<int>(result));
        ctx.cam_id = MLCameraIdentifier_MAIN;
        result = MLCameraConnect(&ctx, &camera_context_);
        if (result != MLResult_Ok) {
            LOGE("MLCameraConnect (MAIN) failed: %d", static_cast<int>(result));
            MLCameraDeInit();
            return false;
        }
    }
    camera_connected_ = true;

    // Device-status callbacks.
    MLCameraDeviceStatusCallbacks dev_cbs = {};
    MLCameraDeviceStatusCallbacksInit(&dev_cbs);
    dev_cbs.on_device_error = [](MLCameraError e, void*) {
        LOGE("Camera error: %d", static_cast<int>(e));
    };
    dev_cbs.on_device_disconnected = [](MLCameraDisconnectReason r, void*) {
        LOGE("Camera disconnected: %d", static_cast<int>(r));
    };
    MLCameraSetDeviceStatusCallbacks(camera_context_, &dev_cbs, this);

    // Capture callbacks.
    MLCameraCaptureCallbacks cap_cbs = {};
    MLCameraCaptureCallbacksInit(&cap_cbs);
    cap_cbs.on_video_buffer_available = RgbCapture::on_video_buffer;
    cap_cbs.on_capture_failed = [](const MLCameraResultExtras*, void*) {
        LOGE("Capture failed");
    };
    cap_cbs.on_capture_aborted = [](void*) { LOGE("Capture aborted"); };
    MLCameraSetCaptureCallbacks(camera_context_, &cap_cbs, this);

    LOGI("RgbCapture init: %dx%d @ %d fps", config_.width, config_.height,
         config_.fps);
    return true;
}

bool RgbCapture::start() {
    if (running_.load()) return true;
    if (!camera_connected_) return false;

    // CV camera tracking handle for per-frame world_from_camera pose queries.
    // Must be created AFTER the perception system is started (which happens in
    // PerceptionCapture::init), otherwise MLCVCameraTrackingCreate returns
    // MLResult_PerceptionSystemNotStarted (12). start() runs after perception
    // init in main.cpp, so this is the correct point.
    if (cv_tracking_handle_ == 0xFFFFFFFFFFFFFFFF &&
        head_handle_        != 0xFFFFFFFFFFFFFFFF) {
        MLResult tres = MLCVCameraTrackingCreate(&cv_tracking_handle_);
        if (tres != MLResult_Ok) {
            LOGE("MLCVCameraTrackingCreate failed: %d — RGB extrinsics disabled",
                 static_cast<int>(tres));
            cv_tracking_handle_ = 0xFFFFFFFFFFFFFFFF;
        } else {
            LOGI("MLCVCameraTrackingCreate OK — RGB extrinsics enabled");
        }
    }

    // Configure video stream.
    MLCameraCaptureConfig capture_cfg = {};
    MLCameraCaptureConfigInit(&capture_cfg);
    capture_cfg.num_streams = 1;
    capture_cfg.stream_config[0].capture_type  = MLCameraCaptureType_Video;
    capture_cfg.stream_config[0].width         = config_.width;
    capture_cfg.stream_config[0].height        = config_.height;
    capture_cfg.stream_config[0].output_format = MLCameraOutputFormat_YUV_420_888;
    capture_cfg.stream_config[0].native_surface_handle = ML_INVALID_HANDLE;

    switch (config_.fps) {
        case 30: capture_cfg.capture_frame_rate = MLCameraCaptureFrameRate_30FPS; break;
        case 60: capture_cfg.capture_frame_rate = MLCameraCaptureFrameRate_60FPS; break;
        default: capture_cfg.capture_frame_rate = MLCameraCaptureFrameRate_15FPS; break;
    }

    MLHandle meta = ML_INVALID_HANDLE;
    MLResult result = MLCameraPrepareCapture(camera_context_, &capture_cfg, &meta);
    if (result != MLResult_Ok) {
        LOGE("MLCameraPrepareCapture failed: %d", static_cast<int>(result));
        return false;
    }
    MLCameraPreCaptureAEAWB(camera_context_);  // non-fatal

    // Write RGB stream config record.
    VrsWriter::RgbConfig rc;
    rc.width   = config_.width;
    rc.height  = config_.height;
    rc.fps     = config_.fps;
    rc.bitrate = config_.use_h264 ? config_.h264_bitrate : 0;
    config_.vrs_writer->write_rgb_config(rc);

    if (config_.use_h264) {
        h264_encoder_ = std::make_unique<H264Encoder>();
        H264Encoder::Config enc_cfg;
        enc_cfg.width          = config_.width;
        enc_cfg.height         = config_.height;
        enc_cfg.fps            = config_.fps;
        enc_cfg.bitrate        = config_.h264_bitrate;
        enc_cfg.iframe_interval = 1;

        // NAL callback: each encoded NAL unit goes straight to VrsWriter.
        // For non-config NALs we look up the pose stashed by copy_loop when
        // that frame was captured (keyed by the PTS µs the encoder is echoing).
        VrsWriter* vrs = config_.vrs_writer;
        std::atomic<uint32_t>* frame_idx = &frame_index_;
        enc_cfg.nal_callback = [vrs, frame_idx, this](
                const uint8_t* data, size_t size,
                bool is_config, bool is_keyframe, int64_t pts_us) {
            int64_t ts_ns = pts_us * 1000;

            VrsWriter::CameraPose pose_copy;
            const VrsWriter::CameraPose* pose_ptr = nullptr;
            if (!is_config) {
                std::lock_guard<std::mutex> lk(pose_map_mutex_);
                auto it = pose_map_.find(pts_us);
                if (it != pose_map_.end()) {
                    pose_copy = it->second;
                    pose_ptr  = &pose_copy;
                }
            }

            vrs->write_rgb_nal(ts_ns, ts_ns,
                               frame_idx->load(),
                               data, size,
                               is_config, is_keyframe,
                               pose_ptr);
        };

        if (!h264_encoder_->init(enc_cfg)) {
            LOGE("H264 encoder init failed — falling back to JPEG");
            h264_encoder_.reset();
            config_.use_h264 = false;
        } else {
            // Pre-allocate the two YUV double-buffers so copy_loop never
            // calls malloc during recording.
            const size_t frame_bytes =
                static_cast<size_t>(config_.width) * config_.height * 3 / 2;
            encode_buf_[0].resize(frame_bytes);
            encode_buf_[1].resize(frame_bytes);
            copy_buf_idx_ = 0;

            encode_thread_running_.store(true);
            encode_thread_ = std::thread(&RgbCapture::encode_loop, this);
            LOGI("RGB using H.264 hardware encoder → VRS (pipelined encode thread)");
        }
    }

    if (!config_.use_h264) {
        // JPEG fallback: encode in WriteQueue, write to VRS.
        VrsWriter* vrs = config_.vrs_writer;
        write_queue_ = std::make_unique<WriteQueue>(
            static_cast<size_t>(config_.queue_depth),
            config_.writer_threads,
            [vrs, this](const WriteTask& task) -> bool {
                // Encode YUV I420 → JPEG in memory.
                std::vector<uint8_t> jpeg;
                auto jpeg_fn = [](void* ctx, void* data, int size) {
                    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                    const uint8_t* b = static_cast<const uint8_t*>(data);
                    v->insert(v->end(), b, b + size);
                };

                const int w = static_cast<int>(task.width);
                const int h = static_cast<int>(task.height);
                const int y_size = w * h;
                const int uv_w = w / 2, uv_h = h / 2;
                const uint8_t* y_ptr = task.data.data();
                const uint8_t* u_ptr = y_ptr + y_size;
                const uint8_t* v_ptr = u_ptr + uv_w * uv_h;

                std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
                for (int r = 0; r < h; ++r) {
                    for (int c = 0; c < w; ++c) {
                        int yv = y_ptr[r * w + c];
                        int uv = u_ptr[(r/2)*uv_w + (c/2)] - 128;
                        int vv = v_ptr[(r/2)*uv_w + (c/2)] - 128;
                        int ri = yv + ((vv * 359) >> 8);
                        int gi = yv - ((uv * 88 + vv * 183) >> 8);
                        int bi = yv + ((uv * 454) >> 8);
                        size_t idx = (r * w + c) * 3;
                        rgb[idx+0] = static_cast<uint8_t>(ri<0?0:ri>255?255:ri);
                        rgb[idx+1] = static_cast<uint8_t>(gi<0?0:gi>255?255:gi);
                        rgb[idx+2] = static_cast<uint8_t>(bi<0?0:bi>255?255:bi);
                    }
                }
                stbi_write_jpg_to_func(jpeg_fn, &jpeg, w, h, 3,
                                       rgb.data(), config_.jpeg_quality);

                if (!jpeg.empty()) {
                    VrsWriter::CameraPose pose;
                    const VrsWriter::CameraPose* pose_ptr = nullptr;
                    if (task.pose_valid) {
                        std::memcpy(pose.position, task.pose_position, sizeof(pose.position));
                        std::memcpy(pose.orientation, task.pose_orientation, sizeof(pose.orientation));
                        pose_ptr = &pose;
                    }
                    vrs->write_rgb_nal(task.timestamp_ns, task.sensor_timestamp_ns,
                                       task.frame_index,
                                       jpeg.data(), jpeg.size(),
                                       false, true,
                                       pose_ptr);
                    return true;
                }
                return false;
            });
    }

    running_.store(true);
    copy_thread_running_.store(true);
    copy_thread_ = std::thread(&RgbCapture::copy_loop, this);

    result = MLCameraCaptureVideoStart(camera_context_);
    if (result != MLResult_Ok) {
        LOGE("MLCameraCaptureVideoStart failed: %d", static_cast<int>(result));
        running_.store(false);
        copy_thread_running_.store(false);
        copy_cv_.notify_all();
        if (copy_thread_.joinable()) copy_thread_.join();
        write_queue_.reset();
        return false;
    }

    LOGI("RgbCapture started");
    return true;
}

void RgbCapture::stop() {
    if (!running_.load() && !camera_connected_) return;

    if (camera_connected_) MLCameraCaptureVideoStop(camera_context_);
    running_.store(false);

    copy_thread_running_.store(false);
    copy_cv_.notify_all();
    if (copy_thread_.joinable()) copy_thread_.join();

    // Drain the encoder thread: it must finish its last frame before we call
    // h264_encoder_->stop() (which sends EOS and waits for the codec to flush).
    if (encode_thread_running_.load()) {
        encode_thread_running_.store(false);
        encode_cv_.notify_all();
        if (encode_thread_.joinable()) encode_thread_.join();
    }

    if (h264_encoder_) {
        h264_encoder_->stop();
        final_written_ = h264_encoder_->frames_written();
        final_dropped_ = h264_encoder_->frames_dropped();
        h264_encoder_.reset();
    } else if (write_queue_) {
        LOGI("RGB: draining write queue (%zu pending)", write_queue_->pending());
        write_queue_->drain_and_stop();
        final_written_ = write_queue_->written();
        final_dropped_ = write_queue_->dropped();
        write_queue_.reset();
    }

    if (camera_connected_) {
        MLCameraDisconnect(camera_context_);
        camera_connected_ = false;
    }
    MLCameraDeInit();

    if (cv_tracking_handle_ != 0xFFFFFFFFFFFFFFFF) {
        MLCVCameraTrackingDestroy(cv_tracking_handle_);
        cv_tracking_handle_ = 0xFFFFFFFFFFFFFFFF;
    }
    {
        std::lock_guard<std::mutex> lk(pose_map_mutex_);
        pose_map_.clear();
    }

    LOGI("RgbCapture stopped: written=%zu dropped=%zu fps=%.1f "
         "callbacks=%u copy_stalls=%u enc_stalls=%u "
         "pose_ok=%u pose_fail=%u",
         frames_written(), frames_dropped(), actual_fps(),
         callback_count_.load(), copy_stalls_.load(), encode_stalls_.load(),
         pose_lookup_ok_.load(), pose_lookup_fail_.load());
}

void RgbCapture::on_video_buffer(const MLCameraOutput* output,
                                  const MLHandle /*meta*/,
                                  const MLCameraResultExtras* extra,
                                  void* data) {
    auto* self = static_cast<RgbCapture*>(data);
    if (!self->running_.load()) return;

    const int64_t cb_enter = now_ns();
    uint32_t cb_num = self->callback_count_.fetch_add(1);
    if (cb_num == 0) self->first_callback_ns_.store(cb_enter);
    self->last_callback_ns_.store(cb_enter);

    const int64_t common_ts = cb_enter;
    const int64_t sensor_ts = extra ? extra->vcam_timestamp : 0;

    if (self->first_frame_ns_.load() == 0) self->first_frame_ns_.store(common_ts);
    self->last_frame_ns_.store(common_ts);

    // Capture intrinsics on first frame.
    if (!self->intrinsics_.valid && extra && extra->intrinsics) {
        std::lock_guard<std::mutex> lock(self->intrinsics_mutex_);
        const auto& p = *extra->intrinsics;
        self->intrinsics_.fx = p.focal_length.x;
        self->intrinsics_.fy = p.focal_length.y;
        self->intrinsics_.cx = p.principal_point.x;
        self->intrinsics_.cy = p.principal_point.y;
        for (int i = 0; i < 5; ++i) self->intrinsics_.distortion[i] = p.distortion[i];
        self->intrinsics_.valid = true;
    }

    // World_from_camera pose for this frame. SDK docs require the query to
    // happen as soon as the frame timestamp is available — cached poses are
    // short-lived. Any failure (PoseNotFound, PerceptionSystemNotStarted at
    // startup) just disables the pose for this frame.
    uint8_t  pose_valid = 0;
    float    pose_pos[3] = {0, 0, 0};
    float    pose_quat[4] = {0, 0, 0, 0};  // x, y, z, w
    if (self->cv_tracking_handle_ != 0xFFFFFFFFFFFFFFFF &&
        self->head_handle_        != 0xFFFFFFFFFFFFFFFF &&
        sensor_ts != 0) {
        MLTransform T = {};
        MLResult pr = MLCVCameraGetFramePose(
            self->cv_tracking_handle_, self->head_handle_,
            MLCVCameraID_ColorCamera, sensor_ts, &T);
        if (pr == MLResult_Ok) {
            pose_valid   = 1;
            pose_pos[0]  = T.position.x;
            pose_pos[1]  = T.position.y;
            pose_pos[2]  = T.position.z;
            pose_quat[0] = T.rotation.x;
            pose_quat[1] = T.rotation.y;
            pose_quat[2] = T.rotation.z;
            pose_quat[3] = T.rotation.w;
            self->pose_lookup_ok_.fetch_add(1);
        } else {
            self->pose_lookup_fail_.fetch_add(1);
        }
    }

    // Hand off to copy thread.
    {
        std::lock_guard<std::mutex> lock(self->copy_mutex_);
        if (self->pending_frame_.ready) self->copy_stalls_.fetch_add(1);
        self->pending_frame_.output      = output;
        self->pending_frame_.common_ts   = common_ts;
        self->pending_frame_.sensor_ts   = sensor_ts;
        self->pending_frame_.frame_index = self->frame_index_.fetch_add(1);
        self->pending_frame_.pose_valid  = pose_valid;
        std::memcpy(self->pending_frame_.pose_position, pose_pos, sizeof(pose_pos));
        std::memcpy(self->pending_frame_.pose_orientation, pose_quat, sizeof(pose_quat));
        self->pending_frame_.ready       = true;
    }
    self->copy_cv_.notify_one();

    const int64_t cb_ns = now_ns() - cb_enter;
    int64_t prev = self->max_callback_ns_.load();
    while (cb_ns > prev &&
           !self->max_callback_ns_.compare_exchange_weak(prev, cb_ns)) {}
}

void RgbCapture::copy_loop() {
    LOGI("RGB copy thread started");

    while (copy_thread_running_.load()) {
        PendingFrame frame;
        {
            std::unique_lock<std::mutex> lock(copy_mutex_);
            copy_cv_.wait(lock, [this] {
                return pending_frame_.ready || !copy_thread_running_.load();
            });
            if (!copy_thread_running_.load() && !pending_frame_.ready) break;
            frame = pending_frame_;
            pending_frame_.ready = false;
        }

        const int64_t copy_start = now_ns();

        const auto& y_plane = frame.output->planes[0];
        const uint32_t w = y_plane.width;
        const uint32_t h = y_plane.height;
        const size_t y_size  = static_cast<size_t>(w) * h;
        const size_t uv_size = (static_cast<size_t>(w) / 2) * (h / 2);
        const size_t total   = y_size + uv_size * 2;

        if (h264_encoder_) {
            // H.264 path: DMA copy into the pre-allocated double buffer; hand
            // off to encode_thread_ immediately without allocating.
            uint8_t* dst = encode_buf_[copy_buf_idx_].data();

            if (y_plane.stride == w) {
                streaming_copy(dst, y_plane.data, y_size);
            } else {
                for (uint32_t row = 0; row < h; ++row)
                    streaming_copy(dst + row * w,
                                   y_plane.data + row * y_plane.stride, w);
            }
            for (int p = 1; p <= 2; ++p) {
                const auto& uv = frame.output->planes[p];
                uint8_t* udst = dst + y_size + (p == 2 ? uv_size : 0);
                const uint32_t uv_w = w / 2, uv_h = h / 2;
                if (uv.pixel_stride == 1 && uv.stride == uv_w) {
                    streaming_copy(udst, uv.data, uv_size);
                } else {
                    for (uint32_t row = 0; row < uv_h; ++row) {
                        const uint8_t* src_row = uv.data + row * uv.stride;
                        for (uint32_t col = 0; col < uv_w; ++col)
                            udst[row * uv_w + col] = src_row[col * uv.pixel_stride];
                    }
                }
            }

            const int64_t copy_ns = now_ns() - copy_start;
            int64_t prev = max_copy_ns_.load();
            while (copy_ns > prev &&
                   !max_copy_ns_.compare_exchange_weak(prev, copy_ns)) {}

            // Stash the pose keyed by the PTS (µs) the encoder will emit with
            // each NAL. Prune entries older than 2 s so the map stays bounded
            // if the encoder ever stalls.
            const int64_t pts_us = frame.common_ts / 1000;
            if (frame.pose_valid) {
                VrsWriter::CameraPose pose;
                std::memcpy(pose.position, frame.pose_position, sizeof(pose.position));
                std::memcpy(pose.orientation, frame.pose_orientation, sizeof(pose.orientation));
                std::lock_guard<std::mutex> lk(pose_map_mutex_);
                pose_map_[pts_us] = pose;
                const int64_t cutoff = pts_us - 2'000'000;
                for (auto it = pose_map_.begin(); it != pose_map_.end(); ) {
                    if (it->first < cutoff) it = pose_map_.erase(it);
                    else                    ++it;
                }
            }

            // Hand off to encode_thread_; swap to the other buffer so the
            // next DMA copy won't touch the buffer encode_thread_ is reading.
            {
                std::lock_guard<std::mutex> lock(encode_mutex_);
                if (pending_encode_.ready) encode_stalls_.fetch_add(1);
                pending_encode_.data         = dst;
                pending_encode_.size         = total;
                pending_encode_.timestamp_us = pts_us;
                pending_encode_.frame_index  = frame.frame_index;
                pending_encode_.ready        = true;
                copy_buf_idx_ ^= 1;
            }
            encode_cv_.notify_one();

            if ((frame.frame_index % 30) == 0) {
                LOGI("RGB copy #%u: copy=%.1fms stalls=%u enc_stalls=%u",
                     frame.frame_index, copy_ns / 1e6,
                     copy_stalls_.load(), encode_stalls_.load());
            }

        } else if (write_queue_) {
            // JPEG fallback: allocate a WriteTask and submit to the queue.
            WriteTask task;
            task.timestamp_ns        = frame.common_ts;
            task.sensor_timestamp_ns = frame.sensor_ts;
            task.width               = w;
            task.height              = h;
            task.frame_index         = frame.frame_index;
            task.stride              = 1;
            task.pose_valid          = frame.pose_valid;
            std::memcpy(task.pose_position, frame.pose_position, sizeof(task.pose_position));
            std::memcpy(task.pose_orientation, frame.pose_orientation, sizeof(task.pose_orientation));
            task.data.resize(total);

            if (y_plane.stride == w) {
                streaming_copy(task.data.data(), y_plane.data, y_size);
            } else {
                for (uint32_t row = 0; row < h; ++row)
                    streaming_copy(task.data.data() + row * w,
                                   y_plane.data + row * y_plane.stride, w);
            }
            for (int p = 1; p <= 2; ++p) {
                const auto& uv = frame.output->planes[p];
                uint8_t* udst = task.data.data() + y_size + (p == 2 ? uv_size : 0);
                const uint32_t uv_w = w / 2, uv_h = h / 2;
                if (uv.pixel_stride == 1 && uv.stride == uv_w) {
                    streaming_copy(udst, uv.data, uv_size);
                } else {
                    for (uint32_t row = 0; row < uv_h; ++row) {
                        const uint8_t* src_row = uv.data + row * uv.stride;
                        for (uint32_t col = 0; col < uv_w; ++col)
                            udst[row * uv_w + col] = src_row[col * uv.pixel_stride];
                    }
                }
            }

            const int64_t copy_ns = now_ns() - copy_start;
            int64_t prev = max_copy_ns_.load();
            while (copy_ns > prev &&
                   !max_copy_ns_.compare_exchange_weak(prev, copy_ns)) {}

            write_queue_->submit(std::move(task));

            if ((frame.frame_index % 30) == 0) {
                LOGI("RGB copy #%u: copy=%.1fms stalls=%u",
                     frame.frame_index, copy_ns / 1e6, copy_stalls_.load());
            }
        }
    }

    LOGI("RGB copy thread exiting");
}

// =============================================================================
// encode_loop — H.264 path only
// Picks up pre-copied YUV frames from pending_encode_ and feeds the hardware
// encoder.  Runs on encode_thread_, decoupled from the DMA copy thread so the
// encoder's ~4 ms per-frame cost doesn't add to the copy budget.
// =============================================================================

void RgbCapture::encode_loop() {
    LOGI("RGB encode thread started");

    while (true) {
        PendingEncode enc;
        {
            std::unique_lock<std::mutex> lock(encode_mutex_);
            encode_cv_.wait(lock, [this] {
                return pending_encode_.ready || !encode_thread_running_.load();
            });
            // Break only when shutting down AND nothing is pending.
            if (!pending_encode_.ready) break;
            enc = pending_encode_;
            pending_encode_.ready = false;
        }

        h264_encoder_->feed(enc.data, enc.size, enc.timestamp_us);
        // NAL units are delivered to VrsWriter via the NalCallback set in start().
    }

    LOGI("RGB encode thread exiting");
}

float RgbCapture::actual_fps() const {
    int64_t first = first_frame_ns_.load();
    int64_t last  = last_frame_ns_.load();
    uint32_t count = frame_index_.load();
    if (count < 2 || last <= first) return 0.0f;
    return static_cast<float>((count - 1) / ((last - first) / 1e9));
}

size_t RgbCapture::frames_written() const {
    return write_queue_ ? write_queue_->written() : final_written_;
}
size_t RgbCapture::frames_dropped() const {
    return write_queue_ ? write_queue_->dropped() : final_dropped_;
}

// JPEG helper stubs — only used in the JPEG fallback WriteQueue path above.
bool RgbCapture::write_jpeg(const uint8_t* rgba, int w, int h,
                             const std::string& path, int q) {
    return stbi_write_jpg(path.c_str(), w, h, 4, rgba, q) != 0;
}
bool RgbCapture::write_yuv_jpeg(const uint8_t* yuv, int w, int h,
                                 const std::string& path, int q) {
    (void)yuv; (void)w; (void)h; (void)path; (void)q;
    return false;  // unused — JPEG fallback encodes inline in WriteQueue lambda
}

}  // namespace ml2
