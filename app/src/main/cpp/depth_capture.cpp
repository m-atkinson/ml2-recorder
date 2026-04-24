#include "depth_capture.h"

#include <android/log.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include <ml_depth_camera.h>

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

DepthCapture::DepthCapture() = default;
DepthCapture::~DepthCapture() { stop(); }

bool DepthCapture::init(const DepthCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("DepthCapture: vrs_writer is null");
        return false;
    }

    MLDepthCameraSettings settings = {};
    MLDepthCameraSettingsInit(&settings);

    if (config_.use_short_range) {
        settings.streams = MLDepthCameraStream_ShortRange;
        settings.stream_configs[MLDepthCameraFrameType_ShortRange].flags =
            MLDepthCameraFlags_DepthImage | MLDepthCameraFlags_Confidence;
        settings.stream_configs[MLDepthCameraFrameType_ShortRange].frame_rate =
            MLDepthCameraFrameRate_5FPS;
    } else {
        settings.streams = MLDepthCameraStream_LongRange;
        settings.stream_configs[MLDepthCameraFrameType_LongRange].flags =
            MLDepthCameraFlags_DepthImage | MLDepthCameraFlags_Confidence;
        settings.stream_configs[MLDepthCameraFrameType_LongRange].frame_rate =
            MLDepthCameraFrameRate_5FPS;
    }

    MLResult result = MLDepthCameraConnect(&settings, &depth_handle_);
    if (result != MLResult_Ok) {
        LOGE("MLDepthCameraConnect failed: %d", static_cast<int>(result));
        return false;
    }

    LOGI("DepthCapture init (%s range)", config_.use_short_range ? "short" : "long");
    return true;
}

bool DepthCapture::start() {
    if (running_.load()) return true;
    running_.store(true);
    capture_thread_ = std::thread(&DepthCapture::capture_loop, this);
    LOGI("DepthCapture started");
    return true;
}

void DepthCapture::stop() {
    if (!running_.load() && depth_handle_ == 0xFFFFFFFFFFFFFFFF) return;
    running_.store(false);
    if (capture_thread_.joinable()) capture_thread_.join();

    if (depth_handle_ != 0xFFFFFFFFFFFFFFFF) {
        MLDepthCameraDisconnect(depth_handle_);
        depth_handle_ = 0xFFFFFFFFFFFFFFFF;
    }
    LOGI("DepthCapture stopped. Frames: %zu", frames_written_.load());
}

void DepthCapture::capture_loop() {
    LOGI("Depth capture thread started");

    MLDepthCameraData depth_data = {};
    MLDepthCameraDataInit(&depth_data);

    while (running_.load()) {
        MLResult result = MLDepthCameraGetLatestDepthData(
            depth_handle_, 500, &depth_data);

        if (result == MLResult_Timeout) continue;
        if (result != MLResult_Ok) {
            LOGE("MLDepthCameraGetLatestDepthData failed: %d",
                 static_cast<int>(result));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        int64_t common_ts = now_ns();

        for (uint8_t i = 0; i < depth_data.frame_count; ++i) {
            const MLDepthCameraFrame& frame = depth_data.frames[i];
            if (!frame.depth_image || !frame.depth_image->data) continue;

            uint32_t w = frame.depth_image->width;
            uint32_t h = frame.depth_image->height;

            // Capture intrinsics on first frame.
            if (actual_width_.load() == 0) {
                actual_width_.store(w);
                actual_height_.store(h);
                intrinsics_.fx = frame.intrinsics.focal_length.x;
                intrinsics_.fy = frame.intrinsics.focal_length.y;
                intrinsics_.cx = frame.intrinsics.principal_point.x;
                intrinsics_.cy = frame.intrinsics.principal_point.y;
                for (int d = 0; d < 5; ++d)
                    intrinsics_.distortion[d] = frame.intrinsics.distortion[d];
                intrinsics_.valid = true;
                LOGI("Depth intrinsics: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                     intrinsics_.fx, intrinsics_.fy,
                     intrinsics_.cx, intrinsics_.cy);
            }

            const float* depth_floats =
                reinterpret_cast<const float*>(frame.depth_image->data);
            const float* conf_floats =
                (frame.confidence && frame.confidence->data)
                ? reinterpret_cast<const float*>(frame.confidence->data)
                : nullptr;

            // Per-frame extrinsic: world-from-camera (ML2 world frame).
            // Quaternion stays SDK-native (x, y, z, w); HDF5 converter reorders.
            VrsWriter::CameraPose pose;
            pose.position[0]    = frame.camera_pose.position.x;
            pose.position[1]    = frame.camera_pose.position.y;
            pose.position[2]    = frame.camera_pose.position.z;
            pose.orientation[0] = frame.camera_pose.rotation.x;
            pose.orientation[1] = frame.camera_pose.rotation.y;
            pose.orientation[2] = frame.camera_pose.rotation.z;
            pose.orientation[3] = frame.camera_pose.rotation.w;

            config_.vrs_writer->write_depth_frame(
                common_ts,
                frame.frame_timestamp,
                w, h,
                frame_index_.fetch_add(1),
                depth_floats,
                conf_floats,
                &pose);

            frames_written_.fetch_add(1);
        }

        MLDepthCameraReleaseDepthData(depth_handle_, &depth_data);
    }

    LOGI("Depth capture thread exiting");
}

}  // namespace ml2
