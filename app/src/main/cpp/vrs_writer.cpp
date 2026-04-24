#include "vrs_writer.h"

#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ── VRS headers ─────────────────────────────────────────────────────────────
// DataLayout.h only forward-declares DataPieceValue/DataPieceArray; include
// their full definitions before any instantiation in this translation unit.
#include <vrs/DataLayout.h>
#include <vrs/DataPieceArray.h>
#include <vrs/DataPieceValue.h>
#include <vrs/DataSource.h>
#include <vrs/RecordFileWriter.h>
#include <vrs/RecordFormat.h>
#include <vrs/Recordable.h>
#include <vrs/StreamId.h>

#define LOG_TAG "ML2VrsWriter"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

// ============================================================================
// RecordableTypeIds
// ============================================================================
// Standard VRS IDs where available; custom (>= 20000) for ML2-specific streams.

static constexpr vrs::RecordableTypeId kRgbTypeId =
    static_cast<vrs::RecordableTypeId>(1203);  // RgbCameraRecordableClass

static constexpr vrs::RecordableTypeId kDepthTypeId =
    static_cast<vrs::RecordableTypeId>(20001);  // custom: ML2 depth camera

static constexpr vrs::RecordableTypeId kWorldCamTypeId =
    static_cast<vrs::RecordableTypeId>(1202);  // SlamCameraData (instance 0/1/2)

static constexpr vrs::RecordableTypeId kHeadPoseTypeId =
    static_cast<vrs::RecordableTypeId>(20002);

static constexpr vrs::RecordableTypeId kEyeTypeId =
    static_cast<vrs::RecordableTypeId>(20003);

static constexpr vrs::RecordableTypeId kHandTypeId =
    static_cast<vrs::RecordableTypeId>(20004);

static constexpr vrs::RecordableTypeId kImuTypeId =
    static_cast<vrs::RecordableTypeId>(1217);  // ImuRecordableClass

static constexpr vrs::RecordableTypeId kAudioTypeId =
    static_cast<vrs::RecordableTypeId>(1204);  // AudioRecordableClass

static constexpr vrs::RecordableTypeId kMeshTypeId =
    static_cast<vrs::RecordableTypeId>(20005);

// Format version used for all data records in this implementation.
// v2: ImageDataLayout gained camera_pose_valid/position/orientation fields.
static constexpr uint32_t kDataVersion = 2;
static constexpr uint32_t kConfigVersion = 1;

// ============================================================================
// DataLayout definitions — one per stream type.
// These are defined as nested structs inside anonymous namespace so they are
// only compiled here (in vrs_writer.cpp) and never exposed in headers.
// ============================================================================

namespace {

// ── IMU ──────────────────────────────────────────────────────────────────────
struct ImuDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t> common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<int64_t> sensor_timestamp_ns{"sensor_timestamp_ns"};
    vrs::DataPieceValue<int32_t> unit_id{"unit_id"};
    vrs::DataPieceArray<float>   accel{"accel", 3};
    vrs::DataPieceArray<float>   gyro{"gyro", 3};
    vrs::AutoDataLayoutEnd end;
};

// ── Head Pose ─────────────────────────────────────────────────────────────────
struct HeadPoseDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t> common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<int64_t> sensor_timestamp_ns{"sensor_timestamp_ns"};
    vrs::DataPieceArray<float>   position{"position", 3};     // x,y,z (metres)
    vrs::DataPieceArray<float>   orientation{"orientation", 4}; // quat w,x,y,z
    vrs::AutoDataLayoutEnd end;
};

// ── Eye Tracking ──────────────────────────────────────────────────────────────
struct EyeDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t> common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<int64_t> sensor_timestamp_ns{"sensor_timestamp_ns"};
    vrs::DataPieceArray<float>   left_origin{"left_origin", 3};
    vrs::DataPieceArray<float>   left_direction{"left_direction", 3};
    vrs::DataPieceArray<float>   right_origin{"right_origin", 3};
    vrs::DataPieceArray<float>   right_direction{"right_direction", 3};
    vrs::DataPieceArray<float>   fixation{"fixation", 3};
    vrs::AutoDataLayoutEnd end;
};

// ── Hand Tracking ─────────────────────────────────────────────────────────────
// 28 keypoints × 2 hands × 3 floats = 168 floats total per hand array.
static constexpr int kHandKpCount = 28;
struct HandDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t>  common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<int64_t>  sensor_timestamp_ns{"sensor_timestamp_ns"};
    vrs::DataPieceValue<uint8_t>  left_valid{"left_valid"};
    vrs::DataPieceValue<uint8_t>  right_valid{"right_valid"};
    vrs::DataPieceValue<float>    left_confidence{"left_confidence"};
    vrs::DataPieceValue<float>    right_confidence{"right_confidence"};
    vrs::DataPieceArray<float>    left_keypoints{"left_keypoints",
                                                  kHandKpCount * 3};
    vrs::DataPieceArray<float>    right_keypoints{"right_keypoints",
                                                   kHandKpCount * 3};
    vrs::AutoDataLayoutEnd end;
};

// ── Image (RGB, Depth, World cam) metadata prefix ────────────────────────────
// Each image data record = this DataLayout followed by the raw image bytes as
// a ContentBlock in the RecordFormat.
struct ImageDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t>  common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<int64_t>  sensor_timestamp_ns{"sensor_timestamp_ns"};
    vrs::DataPieceValue<uint32_t> width{"width"};
    vrs::DataPieceValue<uint32_t> height{"height"};
    vrs::DataPieceValue<uint32_t> frame_index{"frame_index"};
    vrs::DataPieceValue<uint8_t>  is_keyframe{"is_keyframe"};
    vrs::DataPieceValue<uint8_t>  is_config_nal{"is_config_nal"}; // H264 SPS/PPS
    // Per-frame camera pose in the ML2 world frame (world_from_camera).
    // Position is the camera optical-center translation (metres). Orientation
    // is the SDK-native MLQuaternionf layout: [x, y, z, w] (scalar-last).
    // When camera_pose_valid == 0, position & orientation are zero-filled and
    // must be ignored (e.g. RGB config NALs, or frames where the SDK failed to
    // return a pose).
    vrs::DataPieceValue<uint8_t>  camera_pose_valid{"camera_pose_valid"};
    vrs::DataPieceArray<float>    camera_pose_position{"camera_pose_position", 3};
    vrs::DataPieceArray<float>    camera_pose_orientation{"camera_pose_orientation", 4};
    vrs::AutoDataLayoutEnd end;
};

// ── Audio ─────────────────────────────────────────────────────────────────────
struct AudioDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t>  common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<uint32_t> num_samples{"num_samples"};
    vrs::DataPieceValue<uint32_t> num_channels{"num_channels"};
    vrs::AutoDataLayoutEnd end;
};

// ── Mesh ──────────────────────────────────────────────────────────────────────
// Raw geometry — vertices (float*3), indices (uint32), optional normals (float*3).
// Layout carries counts; raw bytes follow as content block.
struct MeshDataLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int64_t>  common_timestamp_ns{"common_timestamp_ns"};
    vrs::DataPieceValue<uint32_t> snapshot_index{"snapshot_index"};
    vrs::DataPieceValue<uint32_t> vertex_count{"vertex_count"};
    vrs::DataPieceValue<uint32_t> index_count{"index_count"};
    vrs::DataPieceValue<uint8_t>  has_normals{"has_normals"};
    vrs::AutoDataLayoutEnd end;
};

// ── Stream config layouts (written once at stream open) ──────────────────────
struct RgbConfigLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int32_t> width{"width"};
    vrs::DataPieceValue<int32_t> height{"height"};
    vrs::DataPieceValue<int32_t> fps{"fps"};
    vrs::DataPieceValue<int32_t> bitrate{"bitrate"};
    // Intrinsics: zero at session start, populated in a second CONFIGURATION
    // record emitted at session close once first-frame SDK values are known.
    vrs::DataPieceValue<float>   fx{"fx"};
    vrs::DataPieceValue<float>   fy{"fy"};
    vrs::DataPieceValue<float>   cx{"cx"};
    vrs::DataPieceValue<float>   cy{"cy"};
    vrs::DataPieceArray<float>   distortion{"distortion", 5};
    vrs::AutoDataLayoutEnd end;
};

struct CameraConfigLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int32_t> width{"width"};
    vrs::DataPieceValue<int32_t> height{"height"};
    vrs::DataPieceValue<float>   fx{"fx"};
    vrs::DataPieceValue<float>   fy{"fy"};
    vrs::DataPieceValue<float>   cx{"cx"};
    vrs::DataPieceValue<float>   cy{"cy"};
    vrs::DataPieceArray<float>   distortion{"distortion", 5};
    vrs::AutoDataLayoutEnd end;
};

struct AudioConfigLayout : vrs::AutoDataLayout {
    vrs::DataPieceValue<int32_t> sample_rate{"sample_rate"};
    vrs::DataPieceValue<int32_t> channels{"channels"};
    vrs::AutoDataLayoutEnd end;
};

// Empty layout for streams that have no configuration fields.
struct EmptyLayout : vrs::AutoDataLayout { vrs::AutoDataLayoutEnd end; };

// ============================================================================
// Generic Recordable — holds a DataLayout instance, emits records.
// ============================================================================

template<typename ConfigLayout, typename DataLayout>
class StreamRecordable : public vrs::Recordable {
public:
    // data_has_content: true for streams whose DATA records have a raw content
    // block after the DataLayout (images, audio, mesh, depth).
    StreamRecordable(vrs::RecordableTypeId type_id,
                     const std::string& stream_flavor,
                     bool data_has_content = false)
        : vrs::Recordable(type_id, stream_flavor) {
        // Register RecordFormats so pyvrs (and other readers) can decode blocks.
        using CT = vrs::ContentType;
        addRecordFormat(vrs::Record::Type::CONFIGURATION, kConfigVersion,
                        vrs::RecordFormat(CT::DATA_LAYOUT), {&config_layout_});
        addRecordFormat(vrs::Record::Type::STATE, kDataVersion,
                        vrs::RecordFormat(CT::DATA_LAYOUT), {&data_layout_});
        if (data_has_content) {
            addRecordFormat(vrs::Record::Type::DATA, kDataVersion,
                            vrs::RecordFormat(vrs::ContentBlock(CT::DATA_LAYOUT),
                                             vrs::ContentBlock(CT::CUSTOM)),
                            {&data_layout_});
        } else {
            addRecordFormat(vrs::Record::Type::DATA, kDataVersion,
                            vrs::RecordFormat(CT::DATA_LAYOUT), {&data_layout_});
        }
    }

    // Write a configuration record with the current state of config_layout.
    void emit_config(double timestamp_s) {
        createRecord(timestamp_s, vrs::Record::Type::CONFIGURATION,
                     kConfigVersion, vrs::DataSource(config_layout_));
    }

    // Write a data record with the current state of data_layout + optional
    // trailing bytes (content block).
    void emit_data(double timestamp_s,
                   const void* extra_data = nullptr,
                   size_t extra_size = 0) {
        if (extra_data && extra_size > 0) {
            createRecord(timestamp_s, vrs::Record::Type::DATA, kDataVersion,
                         vrs::DataSource(data_layout_,
                                         vrs::DataSourceChunk(extra_data, extra_size)));
        } else {
            createRecord(timestamp_s, vrs::Record::Type::DATA, kDataVersion,
                         vrs::DataSource(data_layout_));
        }
    }

    // Required overrides — we don't use STATE records.
    const vrs::Record* createStateRecord() override {
        return createRecord(0.0, vrs::Record::Type::STATE, kDataVersion,
                            vrs::DataSource(data_layout_));
    }
    const vrs::Record* createConfigurationRecord() override {
        return createRecord(0.0, vrs::Record::Type::CONFIGURATION,
                            kConfigVersion, vrs::DataSource(config_layout_));
    }

    ConfigLayout config_layout_;
    DataLayout   data_layout_;
};

// Convenience alias for streams that don't need a content block (pure layout).
template<typename ConfigLayout, typename DataLayout>
using SRec = StreamRecordable<ConfigLayout, DataLayout>;

}  // anonymous namespace

// ============================================================================
// VrsWriter::Impl
// ============================================================================

struct VrsWriter::Impl {
    vrs::RecordFileWriter file_writer;
    bool is_open = false;

    // Stream recordables — one per logical sensor stream.
    // Third arg: data_has_content=true for streams with raw bytes after the DataLayout.
    SRec<RgbConfigLayout,    ImageDataLayout>    rgb{kRgbTypeId,        "ml2/rgb",          true};
    SRec<CameraConfigLayout, ImageDataLayout>    depth{kDepthTypeId,    "ml2/depth",        true};
    SRec<CameraConfigLayout, ImageDataLayout>    wcam0{kWorldCamTypeId, "ml2/world_cam_0",  true};
    SRec<CameraConfigLayout, ImageDataLayout>    wcam1{kWorldCamTypeId, "ml2/world_cam_1",  true};
    SRec<CameraConfigLayout, ImageDataLayout>    wcam2{kWorldCamTypeId, "ml2/world_cam_2",  true};
    SRec<EmptyLayout,        HeadPoseDataLayout> head{kHeadPoseTypeId,  "ml2/head_pose",    false};
    SRec<EmptyLayout,        EyeDataLayout>      eye{kEyeTypeId,        "ml2/eye_tracking", false};
    SRec<EmptyLayout,        HandDataLayout>     hand{kHandTypeId,      "ml2/hand_tracking",false};
    SRec<EmptyLayout,        ImuDataLayout>      imu{kImuTypeId,        "ml2/imu",          false};
    SRec<AudioConfigLayout,  AudioDataLayout>    audio{kAudioTypeId,    "ml2/audio",        true};
    SRec<EmptyLayout,        MeshDataLayout>     mesh{kMeshTypeId,      "ml2/mesh",         true};

    std::map<std::string, std::string> tags;

    // Helper: convert nanoseconds to seconds for VRS timestamps.
    static inline double to_s(int64_t ns) {
        return static_cast<double>(ns) * 1e-9;
    }
};

// ============================================================================
// VrsWriter public implementation
// ============================================================================

VrsWriter::VrsWriter() : impl_(std::make_unique<Impl>()) {}
VrsWriter::~VrsWriter() { close(); }

bool VrsWriter::open(const std::string& filepath) {
    if (impl_->is_open) {
        LOGE("VrsWriter::open called but file already open");
        return false;
    }

    // Register all recordables before creating the file.
    impl_->file_writer.addRecordable(&impl_->rgb);
    impl_->file_writer.addRecordable(&impl_->depth);
    impl_->file_writer.addRecordable(&impl_->wcam0);
    impl_->file_writer.addRecordable(&impl_->wcam1);
    impl_->file_writer.addRecordable(&impl_->wcam2);
    impl_->file_writer.addRecordable(&impl_->head);
    impl_->file_writer.addRecordable(&impl_->eye);
    impl_->file_writer.addRecordable(&impl_->hand);
    impl_->file_writer.addRecordable(&impl_->imu);
    impl_->file_writer.addRecordable(&impl_->audio);
    impl_->file_writer.addRecordable(&impl_->mesh);

    // Apply any tags buffered before open() was called.
    // VRS writes tags at file-creation time, so they must be set before createFileAsync.
    impl_->file_writer.addTags(impl_->tags);

    int status = impl_->file_writer.createFileAsync(filepath);
    if (status != 0) {
        LOGE("VrsWriter: createFileAsync failed (status=%d) path=%s",
             status, filepath.c_str());
        return false;
    }

    // Flush pending records to disk every 200 ms in a background thread.
    impl_->file_writer.autoWriteRecordsAsync(
        []() { return std::numeric_limits<double>::max(); }, 0.2);

    impl_->is_open = true;
    LOGI("VrsWriter: opened %s", filepath.c_str());
    return true;
}

void VrsWriter::close() {
    if (!impl_->is_open) return;

    // Flush all accumulated tags so values set after open() (end_time_ns,
    // frame counts, intrinsics) actually land in the file. Without this
    // every set_tag() call after open() is silently lost.
    impl_->file_writer.addTags(impl_->tags);

    // Write remaining records and close (blocking).
    impl_->file_writer.closeFileAsync();
    impl_->file_writer.waitForFileClosed();
    impl_->is_open = false;
    LOGI("VrsWriter: file closed (%zu tags flushed)", impl_->tags.size());
}

bool VrsWriter::is_open() const { return impl_->is_open; }

void VrsWriter::set_tag(const std::string& key, const std::string& value) {
    // Keep our own mirror so close() can flush everything. Also forward
    // live if the file is already open — addTags() merges, which keeps the
    // single-tag call path symmetrical with the pre-open buffered path.
    impl_->tags[key] = value;
    if (impl_->is_open) {
        impl_->file_writer.addTags({{key, value}});
    }
}

// ============================================================================
// Stream config writers
// ============================================================================

void VrsWriter::write_rgb_config(const RgbConfig& cfg) {
    if (!impl_->is_open) return;
    auto& r = impl_->rgb;
    r.config_layout_.width.set(cfg.width);
    r.config_layout_.height.set(cfg.height);
    r.config_layout_.fps.set(cfg.fps);
    r.config_layout_.bitrate.set(cfg.bitrate);
    r.config_layout_.fx.set(cfg.fx);
    r.config_layout_.fy.set(cfg.fy);
    r.config_layout_.cx.set(cfg.cx);
    r.config_layout_.cy.set(cfg.cy);
    r.config_layout_.distortion.set(cfg.distortion, 5);
    r.emit_config(0.0);
}

void VrsWriter::write_depth_config(const DepthConfig& cfg) {
    if (!impl_->is_open) return;
    auto& r = impl_->depth;
    r.config_layout_.width.set(cfg.width);
    r.config_layout_.height.set(cfg.height);
    r.config_layout_.fx.set(cfg.fx);
    r.config_layout_.fy.set(cfg.fy);
    r.config_layout_.cx.set(cfg.cx);
    r.config_layout_.cy.set(cfg.cy);
    r.config_layout_.distortion.set(cfg.distortion, 5);
    r.emit_config(0.0);
}

void VrsWriter::write_world_cam_config(const WorldCamConfig& cfg) {
    if (!impl_->is_open) return;
    auto* rec = (cfg.cam_index == 0) ? static_cast<SRec<CameraConfigLayout, ImageDataLayout>*>(&impl_->wcam0)
              : (cfg.cam_index == 1) ? &impl_->wcam1
              :                        &impl_->wcam2;
    rec->config_layout_.width.set(cfg.width);
    rec->config_layout_.height.set(cfg.height);
    rec->config_layout_.fx.set(cfg.fx);
    rec->config_layout_.fy.set(cfg.fy);
    rec->config_layout_.cx.set(cfg.cx);
    rec->config_layout_.cy.set(cfg.cy);
    rec->config_layout_.distortion.set(cfg.distortion, 5);
    rec->emit_config(0.0);
}

void VrsWriter::write_audio_config(const AudioConfig& cfg) {
    if (!impl_->is_open) return;
    auto& r = impl_->audio;
    r.config_layout_.sample_rate.set(cfg.sample_rate);
    r.config_layout_.channels.set(cfg.channels);
    r.emit_config(0.0);
}

// ============================================================================
// Data writers
// ============================================================================

// Helper: stage pose fields on any ImageDataLayout. Pass nullptr to clear.
static void set_image_pose(ImageDataLayout& dl, const VrsWriter::CameraPose* pose) {
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (pose) {
        dl.camera_pose_valid.set(1);
        dl.camera_pose_position.set(pose->position, 3);
        dl.camera_pose_orientation.set(pose->orientation, 4);
    } else {
        dl.camera_pose_valid.set(0);
        dl.camera_pose_position.set(zero, 3);
        dl.camera_pose_orientation.set(zero, 4);
    }
}

void VrsWriter::write_rgb_nal(int64_t timestamp_ns, int64_t sensor_ts_ns,
                               uint32_t frame_index,
                               const uint8_t* nal_data, size_t nal_size,
                               bool is_config, bool is_keyframe,
                               const CameraPose* pose) {
    if (!impl_->is_open || !nal_data || nal_size == 0) return;
    auto& r = impl_->rgb;
    auto& dl = r.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    dl.width.set(0);   // encoded in bitstream
    dl.height.set(0);
    dl.frame_index.set(frame_index);
    dl.is_keyframe.set(is_keyframe ? 1 : 0);
    dl.is_config_nal.set(is_config ? 1 : 0);
    // Config NALs (SPS/PPS) have no frame — force pose invalid regardless of caller.
    set_image_pose(dl, is_config ? nullptr : pose);
    r.emit_data(Impl::to_s(timestamp_ns), nal_data, nal_size);
}

void VrsWriter::write_depth_frame(int64_t timestamp_ns, int64_t sensor_ts_ns,
                                   uint32_t width, uint32_t height,
                                   uint32_t frame_index,
                                   const float* depth_m,
                                   const float* confidence,
                                   const CameraPose* pose) {
    if (!impl_->is_open || !depth_m) return;

    const size_t pixel_count = static_cast<size_t>(width) * height;

    // Convert float metres → uint16 millimetres (range 0–65535 mm = 65.5 m max).
    // Confidence is normalised to uint16 if present.
    std::vector<uint16_t> depth_mm(pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
        float mm = depth_m[i] * 1000.0f;
        if (std::isnan(mm) || mm < 0.0f) mm = 0.0f;
        if (mm > 65535.0f) mm = 65535.0f;
        depth_mm[i] = static_cast<uint16_t>(mm);
    }

    std::vector<uint16_t> conf_u16;
    if (confidence) {
        conf_u16.resize(pixel_count);
        float max_val = 0.0f;
        for (size_t i = 0; i < pixel_count; ++i) {
            if (!std::isnan(confidence[i]) && confidence[i] > max_val)
                max_val = confidence[i];
        }
        const float scale = (max_val > 0.0f) ? (65535.0f / max_val) : 0.0f;
        for (size_t i = 0; i < pixel_count; ++i) {
            float v = confidence[i] * scale;
            if (std::isnan(v) || v < 0.0f) v = 0.0f;
            if (v > 65535.0f) v = 65535.0f;
            conf_u16[i] = static_cast<uint16_t>(v);
        }
    }

    // Pack: depth_mm array followed by optional conf_u16 array.
    size_t total_bytes = pixel_count * sizeof(uint16_t)
                       + (conf_u16.empty() ? 0 : pixel_count * sizeof(uint16_t));
    std::vector<uint8_t> payload(total_bytes);
    std::memcpy(payload.data(), depth_mm.data(),
                pixel_count * sizeof(uint16_t));
    if (!conf_u16.empty()) {
        std::memcpy(payload.data() + pixel_count * sizeof(uint16_t),
                    conf_u16.data(), pixel_count * sizeof(uint16_t));
    }

    auto& r = impl_->depth;
    auto& dl = r.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    dl.width.set(width);
    dl.height.set(height);
    dl.frame_index.set(frame_index);
    dl.is_keyframe.set(0);
    dl.is_config_nal.set(conf_u16.empty() ? 0 : 1); // reuse field: 1=has_confidence
    set_image_pose(dl, pose);
    r.emit_data(Impl::to_s(timestamp_ns), payload.data(), payload.size());
}

void VrsWriter::write_world_cam_frame(int cam_index,
                                       int64_t timestamp_ns, int64_t sensor_ts_ns,
                                       uint32_t width, uint32_t height,
                                       uint32_t frame_index,
                                       const uint8_t* jpeg_data, size_t jpeg_size,
                                       const CameraPose* pose) {
    if (!impl_->is_open || !jpeg_data || jpeg_size == 0) return;
    if (cam_index < 0 || cam_index > 2) return;

    auto* rec = (cam_index == 0) ? static_cast<SRec<CameraConfigLayout, ImageDataLayout>*>(&impl_->wcam0)
              : (cam_index == 1) ? &impl_->wcam1
              :                    &impl_->wcam2;

    auto& dl = rec->data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    dl.width.set(width);
    dl.height.set(height);
    dl.frame_index.set(frame_index);
    dl.is_keyframe.set(0);
    dl.is_config_nal.set(0);
    set_image_pose(dl, pose);
    rec->emit_data(Impl::to_s(timestamp_ns), jpeg_data, jpeg_size);
}

void VrsWriter::write_head_pose(int64_t timestamp_ns, int64_t sensor_ts_ns,
                                 float px, float py, float pz,
                                 float qw, float qx, float qy, float qz) {
    if (!impl_->is_open) return;
    auto& dl = impl_->head.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    const float pos[3] = {px, py, pz};
    const float ori[4] = {qw, qx, qy, qz};
    dl.position.set(pos, 3);
    dl.orientation.set(ori, 4);
    impl_->head.emit_data(Impl::to_s(timestamp_ns));
}

void VrsWriter::write_eye_tracking(int64_t timestamp_ns, int64_t sensor_ts_ns,
                                    float lox, float loy, float loz,
                                    float ldx, float ldy, float ldz,
                                    float rox, float roy, float roz,
                                    float rdx, float rdy, float rdz,
                                    float fx, float fy, float fz) {
    if (!impl_->is_open) return;
    auto& dl = impl_->eye.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    const float lo[3] = {lox, loy, loz}, ld[3] = {ldx, ldy, ldz};
    const float ro[3] = {rox, roy, roz}, rd[3] = {rdx, rdy, rdz};
    const float fix[3] = {fx, fy, fz};
    dl.left_origin.set(lo, 3);
    dl.left_direction.set(ld, 3);
    dl.right_origin.set(ro, 3);
    dl.right_direction.set(rd, 3);
    dl.fixation.set(fix, 3);
    impl_->eye.emit_data(Impl::to_s(timestamp_ns));
}

void VrsWriter::write_hand_tracking(int64_t timestamp_ns, int64_t sensor_ts_ns,
                                     bool left_valid, bool right_valid,
                                     float left_conf, float right_conf,
                                     const float* left_kp,
                                     const float* right_kp) {
    if (!impl_->is_open) return;
    auto& dl = impl_->hand.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    dl.left_valid.set(left_valid ? 1 : 0);
    dl.right_valid.set(right_valid ? 1 : 0);
    dl.left_confidence.set(left_conf);
    dl.right_confidence.set(right_conf);
    if (left_kp)  dl.left_keypoints.set(left_kp,  kHandKpCount * 3);
    if (right_kp) dl.right_keypoints.set(right_kp, kHandKpCount * 3);
    impl_->hand.emit_data(Impl::to_s(timestamp_ns));
}

void VrsWriter::write_imu(int64_t timestamp_ns, int64_t sensor_ts_ns,
                           int unit_id,
                           float ax, float ay, float az,
                           float gx, float gy, float gz) {
    if (!impl_->is_open) return;
    auto& dl = impl_->imu.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.sensor_timestamp_ns.set(sensor_ts_ns);
    dl.unit_id.set(unit_id);
    const float a[3] = {ax, ay, az};
    const float g[3] = {gx, gy, gz};
    dl.accel.set(a, 3);
    dl.gyro.set(g, 3);
    impl_->imu.emit_data(Impl::to_s(timestamp_ns));
}

void VrsWriter::write_audio(int64_t timestamp_ns,
                             const int16_t* samples, uint32_t num_samples,
                             uint32_t num_channels) {
    if (!impl_->is_open || !samples || num_samples == 0) return;
    auto& r = impl_->audio;
    auto& dl = r.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.num_samples.set(num_samples);
    dl.num_channels.set(num_channels);
    const size_t byte_size = static_cast<size_t>(num_samples)
                           * num_channels * sizeof(int16_t);
    r.emit_data(Impl::to_s(timestamp_ns),
                reinterpret_cast<const uint8_t*>(samples), byte_size);
}

void VrsWriter::write_mesh(int64_t timestamp_ns,
                            uint32_t snapshot_index,
                            uint32_t vertex_count, uint32_t index_count,
                            const float* vertices,
                            const uint32_t* indices,
                            const float* normals) {
    if (!impl_->is_open || !vertices || !indices) return;

    auto& r = impl_->mesh;
    auto& dl = r.data_layout_;
    dl.common_timestamp_ns.set(timestamp_ns);
    dl.snapshot_index.set(snapshot_index);
    dl.vertex_count.set(vertex_count);
    dl.index_count.set(index_count);
    dl.has_normals.set(normals ? 1 : 0);

    // Pack: vertices | indices | optional normals
    const size_t vert_bytes = static_cast<size_t>(vertex_count) * 3 * sizeof(float);
    const size_t idx_bytes  = static_cast<size_t>(index_count)  * sizeof(uint32_t);
    const size_t norm_bytes = normals ? vert_bytes : 0;
    std::vector<uint8_t> payload(vert_bytes + idx_bytes + norm_bytes);

    uint8_t* p = payload.data();
    std::memcpy(p, vertices, vert_bytes); p += vert_bytes;
    std::memcpy(p, indices,  idx_bytes);  p += idx_bytes;
    if (normals) std::memcpy(p, normals, norm_bytes);

    r.emit_data(Impl::to_s(timestamp_ns), payload.data(), payload.size());
}

}  // namespace ml2
