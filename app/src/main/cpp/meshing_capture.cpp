#include "meshing_capture.h"

#include <android/log.h>

#include <chrono>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <ml_meshing2.h>
#include <ml_head_tracking.h>
#include <ml_perception.h>
#include <ml_snapshot.h>

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

MeshingCapture::MeshingCapture() = default;
MeshingCapture::~MeshingCapture() { stop(); }

bool MeshingCapture::init(const MeshingCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("MeshingCapture: vrs_writer is null");
        return false;
    }

    MLMeshingSettings settings = {};
    MLMeshingInitSettings(&settings);
    settings.fill_hole_length             = 0.5f;
    settings.disconnected_component_area  = 0.1f;
    if (config_.compute_normals)     settings.flags |= MLMeshingFlags_ComputeNormals;
    if (config_.compute_confidence)  settings.flags |= MLMeshingFlags_ComputeConfidence;
    settings.flags |= MLMeshingFlags_RemoveMeshSkirt;

    MLResult result = MLMeshingCreateClient(&meshing_client_, &settings);
    if (result != MLResult_Ok) {
        LOGE("MLMeshingCreateClient failed: %d", static_cast<int>(result));
        return false;
    }

    LOGI("MeshingCapture init: poll=%.1fs bounds=%.1fm normals=%d lod=%d",
         config_.poll_interval_s, config_.bounds_extents_m,
         config_.compute_normals, config_.lod);
    return true;
}

bool MeshingCapture::start() {
    if (running_.load()) return true;
    running_.store(true);
    capture_thread_ = std::thread(&MeshingCapture::capture_loop, this);
    return true;
}

void MeshingCapture::stop() {
    if (!running_.load() && meshing_client_ == 0xFFFFFFFFFFFFFFFF) return;
    running_.store(false);
    if (capture_thread_.joinable()) capture_thread_.join();
    if (meshing_client_ != 0xFFFFFFFFFFFFFFFF) {
        MLMeshingDestroyClient(meshing_client_);
        meshing_client_ = 0xFFFFFFFFFFFFFFFF;
    }
    LOGI("MeshingCapture stopped: %u snapshots, %llu total vertices",
         snapshots_written_.load(), (unsigned long long)total_vertices_.load());
}

void MeshingCapture::capture_loop() {
    LOGI("Meshing capture thread started");

    MLMeshingLOD lod;
    switch (config_.lod) {
        case 0:  lod = MLMeshingLOD_Minimum; break;
        case 2:  lod = MLMeshingLOD_Maximum; break;
        default: lod = MLMeshingLOD_Medium;  break;
    }

    MLHandle head_tracker = ML_INVALID_HANDLE;
    MLResult ht_result = MLHeadTrackingCreate(&head_tracker);
    if (ht_result != MLResult_Ok || head_tracker == ML_INVALID_HANDLE) {
        LOGE("MLHeadTrackingCreate failed: %d — will query mesh at origin",
             static_cast<int>(ht_result));
    }
    MLHeadTrackingStaticData head_static = {};
    if (head_tracker != ML_INVALID_HANDLE) {
        MLResult sd_result = MLHeadTrackingGetStaticData(head_tracker, &head_static);
        if (sd_result != MLResult_Ok)
            LOGE("MLHeadTrackingGetStaticData failed: %d", static_cast<int>(sd_result));
    }

    const int64_t poll_ns =
        static_cast<int64_t>(config_.poll_interval_s * 1e9);

    while (running_.load()) {
        const int64_t loop_start = now_ns();

        // ── Get head position for query bounds ──────────────────────────────
        MLVec3f head_pos = {0, 0, 0};
        {
            MLSnapshot* snap = nullptr;
            if (MLPerceptionGetSnapshot(&snap) == MLResult_Ok && snap) {
                MLTransform t = {};
                MLResult tr = MLSnapshotGetTransform(snap, &head_static.coord_frame_head, &t);
                if (tr == MLResult_Ok) head_pos = t.position;
                MLPerceptionReleaseSnapshot(snap);
            }
        }

        // ── Request mesh info ───────────────────────────────────────────────
        MLMeshingExtents extents = {};
        extents.center   = head_pos;
        extents.rotation = {0, 0, 0, 1};
        extents.extents.x = extents.extents.y = extents.extents.z =
            config_.bounds_extents_m;

        MLHandle info_req = ML_INVALID_HANDLE;
        MLResult result = MLMeshingRequestMeshInfo(meshing_client_, &extents,
                                                    &info_req);
        if (result != MLResult_Ok) {
            LOGE("MLMeshingRequestMeshInfo failed: %d", static_cast<int>(result));
            goto sleep_and_continue;
        }

        {
            // ── Poll for mesh info ──────────────────────────────────────────
            MLMeshingMeshInfo mesh_info = {};
            bool got_info = false;
            for (int a = 0; a < 100 && running_.load(); ++a) {
                result = MLMeshingGetMeshInfoResult(meshing_client_, info_req,
                                                    &mesh_info);
                if (result == MLResult_Ok) { got_info = true; break; }
                if (result == MLResult_Pending) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                LOGE("MLMeshingGetMeshInfoResult error: %d", static_cast<int>(result));
                break;
            }

            if (!got_info) {
                LOGE("MLMeshingGetMeshInfoResult timed out or failed (data_count would be %u)",
                     mesh_info.data_count);
                MLMeshingFreeResource(meshing_client_, &info_req);
                goto sleep_and_continue;
            }

            LOGI("Mesh info: %u blocks", mesh_info.data_count);

            if (mesh_info.data_count == 0) {
                MLMeshingFreeResource(meshing_client_, &info_req);
                goto sleep_and_continue;
            }

            // ── Request mesh data for new/updated blocks ────────────────────
            // Build block_reqs BEFORE freeing info_req — mesh_info.data is
            // owned by the SDK and freed along with the info request handle.
            std::vector<MLMeshingBlockRequest> block_reqs;
            for (uint32_t i = 0; i < mesh_info.data_count; ++i) {
                const auto& b = mesh_info.data[i];
                if (b.state == MLMeshingMeshState_New ||
                    b.state == MLMeshingMeshState_Updated) {
                    block_reqs.push_back({b.id, lod});
                }
            }

            // Safe to free now that block_reqs is built.
            MLMeshingFreeResource(meshing_client_, &info_req);

            LOGI("Mesh blocks: %zu new/updated of %u total",
                 block_reqs.size(), mesh_info.data_count);

            if (block_reqs.empty()) goto sleep_and_continue;

            MLMeshingMeshRequest mesh_request = {};
            mesh_request.request_count = static_cast<int>(block_reqs.size());
            mesh_request.data          = block_reqs.data();

            MLHandle mesh_handle = ML_INVALID_HANDLE;
            result = MLMeshingRequestMesh(meshing_client_, &mesh_request,
                                           &mesh_handle);
            if (result != MLResult_Ok) {
                LOGE("MLMeshingRequestMesh failed: %d", static_cast<int>(result));
                goto sleep_and_continue;
            }

            // ── Poll for mesh data ──────────────────────────────────────────
            MLMeshingMesh mesh_data = {};
            bool got_mesh = false;
            for (int a = 0; a < 200 && running_.load(); ++a) {
                result = MLMeshingGetMeshResult(meshing_client_, mesh_handle,
                                                &mesh_data);
                if (result == MLResult_Ok) { got_mesh = true; break; }
                if (result == MLResult_Pending) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                LOGE("MLMeshingGetMeshResult error: %d", static_cast<int>(result));
                break;
            }

            if (!got_mesh) {
                LOGE("MLMeshingGetMeshResult timed out or failed");
                MLMeshingFreeResource(meshing_client_, &mesh_handle);
                goto sleep_and_continue;
            }

            {
                // ── Merge blocks into flat arrays and write VRS record ──────
                uint32_t total_verts = 0, total_idx = 0, valid_blocks = 0;
                for (uint32_t i = 0; i < mesh_data.data_count; ++i) {
                    const auto& b = mesh_data.data[i];
                    if (b.vertex_count > 0 && b.index_count > 0) {
                        total_verts += b.vertex_count;
                        total_idx   += b.index_count;
                        valid_blocks++;
                    }
                }

                LOGI("Mesh result: %u blocks, %u verts, %u tris",
                     valid_blocks, total_verts, total_idx / 3);

                if (total_verts > 0) {
                    std::vector<float>    verts(total_verts * 3);
                    std::vector<uint32_t> indices(total_idx);
                    std::vector<float>    normals;
                    if (config_.compute_normals) normals.resize(total_verts * 3);

                    uint32_t v_off = 0, i_off = 0;
                    for (uint32_t b = 0; b < mesh_data.data_count; ++b) {
                        const auto& blk = mesh_data.data[b];
                        if (!blk.vertex_count || !blk.index_count) continue;

                        std::memcpy(verts.data() + v_off * 3,
                                    blk.vertex,
                                    blk.vertex_count * 3 * sizeof(float));
                        if (config_.compute_normals && blk.normal) {
                            std::memcpy(normals.data() + v_off * 3,
                                        blk.normal,
                                        blk.vertex_count * 3 * sizeof(float));
                        }
                        for (uint32_t j = 0; j < blk.index_count; ++j)
                            indices[i_off + j] = blk.index[j] + v_off;

                        v_off += blk.vertex_count;
                        i_off += blk.index_count;
                    }

                    uint32_t snap_idx = snapshot_index_.fetch_add(1);
                    config_.vrs_writer->write_mesh(
                        now_ns(), snap_idx,
                        total_verts, total_idx,
                        verts.data(), indices.data(),
                        normals.empty() ? nullptr : normals.data());

                    snapshots_written_.fetch_add(1);
                    total_vertices_.fetch_add(total_verts);
                }

                // Free mesh data AFTER we've finished reading from mesh_data.data.
                MLMeshingFreeResource(meshing_client_, &mesh_handle);
            }
        }

    sleep_and_continue: {
            int64_t sleep_until = loop_start + poll_ns;
            while (running_.load() && now_ns() < sleep_until)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    if (head_tracker != ML_INVALID_HANDLE)
        MLHeadTrackingDestroy(head_tracker);

    LOGI("Meshing capture thread exiting");
}

}  // namespace ml2
