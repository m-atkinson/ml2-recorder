#include "world_camera_capture.h"

#include <android/log.h>

#include <chrono>
#include <cstring>
#include <vector>

#include <ml_world_camera.h>

// stb_image_write — implementation in rgb_capture.cpp
#include "stb_image_write.h"

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

int WorldCameraCapture::camera_id_to_index(uint32_t cam_id) const {
    if (cam_id & MLWorldCameraIdentifier_Left)   return 0;
    if (cam_id & MLWorldCameraIdentifier_Right)  return 1;
    if (cam_id & MLWorldCameraIdentifier_Center) return 2;
    return -1;
}

WorldCameraCapture::WorldCameraCapture() = default;
WorldCameraCapture::~WorldCameraCapture() { stop(); }

bool WorldCameraCapture::init(const WorldCameraCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("WorldCameraCapture: vrs_writer is null");
        return false;
    }

    uint32_t cameras = 0;
    if (config_.enable_left)   cameras |= MLWorldCameraIdentifier_Left;
    if (config_.enable_right)  cameras |= MLWorldCameraIdentifier_Right;
    if (config_.enable_center) cameras |= MLWorldCameraIdentifier_Center;

    if (cameras == 0) {
        LOGE("No world cameras enabled");
        return false;
    }

    MLWorldCameraSettings settings = {};
    MLWorldCameraSettingsInit(&settings);
    settings.mode    = MLWorldCameraMode_NormalExposure;
    settings.cameras = cameras;

    MLResult result = MLWorldCameraConnect(&settings, &world_cam_handle_);
    if (result != MLResult_Ok) {
        LOGE("MLWorldCameraConnect failed: %d", static_cast<int>(result));
        return false;
    }

    LOGI("World cameras connected (mask=0x%x)", cameras);
    return true;
}

bool WorldCameraCapture::start() {
    if (running_.load()) return true;

    // Create one independent WriteQueue per camera.  Each queue's encode fn
    // JPEG-encodes the raw frame and writes it to VrsWriter.  Independent
    // queues prevent one camera's burst from starving the others.
    for (int cam = 0; cam < 3; ++cam) {
        bool enabled = (cam == 0 && config_.enable_left)  ||
                       (cam == 1 && config_.enable_right) ||
                       (cam == 2 && config_.enable_center);
        if (!enabled) continue;

        write_queue_[cam] = std::make_unique<WriteQueue>(
            static_cast<size_t>(config_.queue_depth),
            config_.writer_threads,
            [this, cam](const WriteTask& task) -> bool {
                std::vector<uint8_t> jpeg;
                if (!encode_jpeg(task.data.data(), task.width, task.height,
                                  task.stride, config_.jpeg_quality, jpeg)) {
                    return false;
                }
                VrsWriter::CameraPose pose;
                const VrsWriter::CameraPose* pose_ptr = nullptr;
                if (task.pose_valid) {
                    std::memcpy(pose.position, task.pose_position, sizeof(pose.position));
                    std::memcpy(pose.orientation, task.pose_orientation, sizeof(pose.orientation));
                    pose_ptr = &pose;
                }
                config_.vrs_writer->write_world_cam_frame(
                    cam,
                    task.timestamp_ns,
                    task.sensor_timestamp_ns,
                    task.width, task.height,
                    task.frame_index,
                    jpeg.data(), jpeg.size(),
                    pose_ptr);
                frames_written_[cam].fetch_add(1);
                return true;
            });
    }

    running_.store(true);
    capture_thread_ = std::thread(&WorldCameraCapture::capture_loop, this);
    LOGI("WorldCameraCapture started (3 independent queues, %d threads each)",
         config_.writer_threads);
    return true;
}

void WorldCameraCapture::stop() {
    if (!running_.load() && world_cam_handle_ == 0xFFFFFFFFFFFFFFFF) return;

    running_.store(false);
    if (capture_thread_.joinable()) capture_thread_.join();

    for (int i = 0; i < 3; ++i) {
        if (write_queue_[i]) {
            LOGI("World cam %d: draining queue (%zu pending)",
                 i, write_queue_[i]->pending());
            write_queue_[i]->drain_and_stop();
            final_dropped_[i] = write_queue_[i]->dropped();
            write_queue_[i].reset();
        }
    }

    if (world_cam_handle_ != 0xFFFFFFFFFFFFFFFF) {
        MLWorldCameraDisconnect(world_cam_handle_);
        world_cam_handle_ = 0xFFFFFFFFFFFFFFFF;
    }

    LOGI("WorldCameraCapture stopped: cam0=%zu cam1=%zu cam2=%zu, "
         "dropped=%zu/%zu/%zu",
         frames_written_[0].load(), frames_written_[1].load(),
         frames_written_[2].load(),
         final_dropped_[0], final_dropped_[1], final_dropped_[2]);
}

size_t WorldCameraCapture::frames_written(int cam) const {
    if (cam < 0 || cam >= 3) return 0;
    return frames_written_[cam].load();
}

size_t WorldCameraCapture::frames_dropped() const {
    size_t total = 0;
    for (int i = 0; i < 3; ++i) {
        total += write_queue_[i] ? write_queue_[i]->dropped() : final_dropped_[i];
    }
    return total;
}

void WorldCameraCapture::capture_loop() {
    LOGI("World camera capture thread started");

    MLWorldCameraData cam_data_storage = {};
    MLWorldCameraDataInit(&cam_data_storage);

    while (running_.load()) {
        MLWorldCameraData* cam_data = &cam_data_storage;
        MLResult result = MLWorldCameraGetLatestWorldCameraData(
            world_cam_handle_, 10, &cam_data);

        if (result == MLResult_Timeout) continue;
        if (result != MLResult_Ok || !cam_data) {
            if (result != MLResult_Ok)
                LOGE("MLWorldCameraGetLatestWorldCameraData failed: %d",
                     static_cast<int>(result));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        int64_t common_ts = now_ns();

        for (uint8_t i = 0; i < cam_data->frame_count; ++i) {
            const MLWorldCameraFrame& frame = cam_data->frames[i];
            int idx = camera_id_to_index(frame.id);
            if (idx < 0 || !write_queue_[idx]) continue;

            if (actual_width_.load() == 0) {
                actual_width_.store(frame.frame_buffer.width);
                actual_height_.store(frame.frame_buffer.height);
            }

            if (!intrinsics_[idx].valid) {
                intrinsics_[idx].fx = frame.intrinsics.focal_length.x;
                intrinsics_[idx].fy = frame.intrinsics.focal_length.y;
                intrinsics_[idx].cx = frame.intrinsics.principal_point.x;
                intrinsics_[idx].cy = frame.intrinsics.principal_point.y;
                intrinsics_[idx].distortion[0] = frame.intrinsics.radial_distortion[0];
                intrinsics_[idx].distortion[1] = frame.intrinsics.radial_distortion[1];
                intrinsics_[idx].distortion[2] = frame.intrinsics.tangential_distortion[0];
                intrinsics_[idx].distortion[3] = frame.intrinsics.tangential_distortion[1];
                intrinsics_[idx].distortion[4] = frame.intrinsics.radial_distortion[2];
                intrinsics_[idx].valid = true;
                LOGI("World cam %d intrinsics: fx=%.2f fy=%.2f",
                     idx, intrinsics_[idx].fx, intrinsics_[idx].fy);
            }

            WriteTask task;
            task.timestamp_ns        = common_ts;
            task.sensor_timestamp_ns = frame.timestamp;
            task.width               = frame.frame_buffer.width;
            task.height              = frame.frame_buffer.height;
            task.stride              = frame.frame_buffer.stride;
            task.stream_id           = idx;
            task.frame_index         = frame_index_[idx].fetch_add(1);

            // Per-frame extrinsic: world-from-camera in the ML2 world frame.
            // Quaternion stays SDK-native (x, y, z, w); HDF5 converter reorders.
            task.pose_valid          = 1;
            task.pose_position[0]    = frame.camera_pose.position.x;
            task.pose_position[1]    = frame.camera_pose.position.y;
            task.pose_position[2]    = frame.camera_pose.position.z;
            task.pose_orientation[0] = frame.camera_pose.rotation.x;
            task.pose_orientation[1] = frame.camera_pose.rotation.y;
            task.pose_orientation[2] = frame.camera_pose.rotation.z;
            task.pose_orientation[3] = frame.camera_pose.rotation.w;

            size_t data_size =
                static_cast<size_t>(frame.frame_buffer.stride) *
                frame.frame_buffer.height;
            task.data.resize(data_size);
            std::memcpy(task.data.data(), frame.frame_buffer.data, data_size);

            write_queue_[idx]->submit(std::move(task));
        }

        MLWorldCameraReleaseCameraData(world_cam_handle_, cam_data);
    }

    LOGI("World camera capture thread exiting");
}

// ---------------------------------------------------------------------------
// encode_jpeg — JPEG-encode 8-bit grayscale into |out| (thread-safe).
// ---------------------------------------------------------------------------

// stbi_write_jpg callback that appends to a std::vector<uint8_t>.
static void jpeg_write_func(void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<uint8_t>*>(ctx);
    const auto* bytes = static_cast<const uint8_t*>(data);
    out->insert(out->end(), bytes, bytes + size);
}

bool WorldCameraCapture::encode_jpeg(const uint8_t* data,
                                      uint32_t width, uint32_t height,
                                      uint32_t stride, int quality,
                                      std::vector<uint8_t>& out) {
    out.clear();
    // stbi_write_jpg doesn't support stride — pack if needed.
    if (stride != width) {
        std::vector<uint8_t> packed(static_cast<size_t>(width) * height);
        for (uint32_t row = 0; row < height; ++row)
            std::memcpy(packed.data() + row * width, data + row * stride, width);
        return stbi_write_jpg_to_func(jpeg_write_func, &out,
                                      static_cast<int>(width),
                                      static_cast<int>(height),
                                      1, packed.data(), quality) != 0;
    }
    return stbi_write_jpg_to_func(jpeg_write_func, &out,
                                  static_cast<int>(width),
                                  static_cast<int>(height),
                                  1, data, quality) != 0;
}

}  // namespace ml2
