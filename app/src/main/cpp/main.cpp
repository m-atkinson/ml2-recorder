#include <android_native_app_glue.h>
#include <android/log.h>
#include <jni.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "audio_capture.h"
#include "capture_profile.h"
#include "depth_capture.h"
#include "imu_capture.h"
#include "meshing_capture.h"
#include "perception_capture.h"
#include "rgb_capture.h"
#include "session.h"
#include "storage_monitor.h"
#include "thermal_monitor.h"
#include "timestamp.h"
#include "vrs_writer.h"
#include "world_camera_capture.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// JNI helpers — read Intent extras
// ---------------------------------------------------------------------------

static int get_intent_int(struct android_app* app, const char* key,
                          int default_val) {
    if (!app->activity || !app->activity->vm) return default_val;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return default_val;

    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jobject intent   = env->CallObjectMethod(
        activity, env->GetMethodID(ac, "getIntent", "()Landroid/content/Intent;"));
    jclass ic = env->GetObjectClass(intent);
    jstring jkey = env->NewStringUTF(key);
    int value = env->CallIntMethod(
        intent,
        env->GetMethodID(ic, "getIntExtra", "(Ljava/lang/String;I)I"),
        jkey, default_val);
    env->DeleteLocalRef(jkey);
    app->activity->vm->DetachCurrentThread();
    return value;
}

static bool get_intent_bool(struct android_app* app, const char* key, bool def) {
    return get_intent_int(app, key, def ? 1 : 0) != 0;
}

static std::string get_intent_string(struct android_app* app, const char* key,
                                     const std::string& def) {
    if (!app->activity || !app->activity->vm) return def;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return def;

    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jobject intent   = env->CallObjectMethod(
        activity, env->GetMethodID(ac, "getIntent", "()Landroid/content/Intent;"));
    jclass  ic   = env->GetObjectClass(intent);
    jstring jkey = env->NewStringUTF(key);
    jstring jval = static_cast<jstring>(env->CallObjectMethod(
        intent,
        env->GetMethodID(ic, "getStringExtra",
                         "(Ljava/lang/String;)Ljava/lang/String;"),
        jkey));
    std::string result = def;
    if (jval) {
        const char* s = env->GetStringUTFChars(jval, nullptr);
        if (s) { result = s; env->ReleaseStringUTFChars(jval, s); }
    }
    env->DeleteLocalRef(jkey);
    app->activity->vm->DetachCurrentThread();
    return result;
}

static std::string get_external_files_dir(struct android_app* app) {
    if (!app->activity || !app->activity->vm) return "";
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return "";

    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jobject file_obj = env->CallObjectMethod(
        activity,
        env->GetMethodID(ac, "getExternalFilesDir",
                         "(Ljava/lang/String;)Ljava/io/File;"),
        nullptr);
    std::string result;
    if (file_obj) {
        jclass  fc   = env->GetObjectClass(file_obj);
        jstring path = static_cast<jstring>(env->CallObjectMethod(
            file_obj,
            env->GetMethodID(fc, "getAbsolutePath", "()Ljava/lang/String;")));
        const char* s = env->GetStringUTFChars(path, nullptr);
        result = s;
        env->ReleaseStringUTFChars(path, s);
    }
    app->activity->vm->DetachCurrentThread();
    return result;
}

static void show_toast(struct android_app* app, const char* msg) {
    if (!app->activity || !app->activity->vm) return;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jmethodID m = env->GetMethodID(ac, "showToast", "(Ljava/lang/String;)V");
    if (m) {
        jstring jmsg = env->NewStringUTF(msg);
        env->CallVoidMethod(activity, m, jmsg);
        env->DeleteLocalRef(jmsg);
    }
    app->activity->vm->DetachCurrentThread();
}

static void play_start_tone(struct android_app* app) {
    if (!app->activity || !app->activity->vm) return;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jmethodID m = env->GetMethodID(ac, "playStartTone", "()V");
    if (m) env->CallVoidMethod(activity, m);
    app->activity->vm->DetachCurrentThread();
}

static void play_stop_tone(struct android_app* app) {
    if (!app->activity || !app->activity->vm) return;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    jmethodID m = env->GetMethodID(ac, "playStopTone", "()V");
    if (m) env->CallVoidMethod(activity, m);
    app->activity->vm->DetachCurrentThread();
}

static void request_permissions(struct android_app* app) {
    if (!app->activity || !app->activity->vm) return;
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    if (!env) return;
    jobject activity = app->activity->clazz;
    jclass  ac       = env->GetObjectClass(activity);
    const char* perms[] = {
        // Standard Android permissions
        "android.permission.CAMERA",
        "android.permission.RECORD_AUDIO",
        "android.permission.WRITE_EXTERNAL_STORAGE",
        "android.permission.READ_EXTERNAL_STORAGE",
        // ML2-specific permissions (must be requested at runtime AND granted
        // via device Settings > Privacy for privileged permissions like
        // SPATIAL_MAPPING and EYE_TRACKING).
        "com.magicleap.permission.PERCEPTION",
        "com.magicleap.permission.EYE_TRACKING",
        "com.magicleap.permission.HAND_TRACKING",
        "com.magicleap.permission.DEPTH_CAMERA",
        "com.magicleap.permission.SPATIAL_MAPPING",
    };
    const int n_perms = static_cast<int>(sizeof(perms) / sizeof(perms[0]));
    jclass string_class = env->FindClass("java/lang/String");
    jobjectArray arr = env->NewObjectArray(n_perms, string_class, nullptr);
    for (int i = 0; i < n_perms; ++i)
        env->SetObjectArrayElement(arr, i, env->NewStringUTF(perms[i]));
    jmethodID m = env->GetMethodID(ac, "requestPermissions",
                                    "([Ljava/lang/String;I)V");
    env->CallVoidMethod(activity, m, arr, 1001);
    app->activity->vm->DetachCurrentThread();
}

// ---------------------------------------------------------------------------
// App event handler
// ---------------------------------------------------------------------------

static bool window_ready = false;

static void handle_cmd(struct android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:  window_ready = true;  break;
        case APP_CMD_TERM_WINDOW:  window_ready = false; break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// android_main
// ---------------------------------------------------------------------------

void android_main(struct android_app* app) {
    LOGI("ML2 Recorder (VRS) starting, ts=%lld", (long long)ml2::now_ns());

    app->onAppCmd = handle_cmd;

    int duration_s = get_intent_int(app, "duration", 30);
    std::string profile_name = get_intent_string(app, "profile", "full_quality");

    ml2::CaptureProfile profile = ml2::CaptureProfile::from_name(profile_name);

    bool enable_rgb     = get_intent_bool(app, "rgb",           profile.rgb);
    bool enable_depth   = get_intent_bool(app, "depth",         profile.depth);
    bool enable_world   = get_intent_bool(app, "world_cams",    profile.world_cams);
    bool enable_head    = get_intent_bool(app, "head_pose",     profile.head_pose);
    bool enable_eye     = get_intent_bool(app, "eye_tracking",  profile.eye_tracking);
    bool enable_hand    = get_intent_bool(app, "hand_tracking", profile.hand_tracking);
    bool enable_imu     = get_intent_bool(app, "imu",           profile.imu);
    bool enable_audio   = get_intent_bool(app, "audio",         profile.audio);
    bool enable_meshing = get_intent_bool(app, "meshing",       profile.meshing);

    LOGI("Profile: %s  duration: %ds", profile.name.c_str(), duration_s);

    request_permissions(app);

    for (int i = 0; i < 60 && !app->destroyRequested; ++i) {
        int events;
        android_poll_source* source;
        while (ALooper_pollOnce(50, nullptr, &events,
                                reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
        }
        if (window_ready) break;
    }

    // ------------------------------------------------------------------
    // 1. Create session (directory for the VRS filepath, not data files)
    // ------------------------------------------------------------------
    std::string base_dir = get_external_files_dir(app);
    if (base_dir.empty()) base_dir = "/sdcard/ml2_recordings";
    base_dir += "/recordings";

    ml2::Session session;
    if (!session.create(base_dir)) {
        LOGE("Failed to create session directory");
        return;
    }

    const std::string vrs_path = session.vrs_path();
    LOGI("VRS output: %s", vrs_path.c_str());

    // ------------------------------------------------------------------
    // 2. Open VrsWriter (single output file for all streams)
    // VRS file-level tags must be set BEFORE createFileAsync, so we
    // buffer them on the VrsWriter and then call open().
    // ------------------------------------------------------------------
    ml2::VrsWriter vrs;

    // Write session-level metadata as VRS file tags (before open).
    const int64_t start_time_ns = ml2::now_ns();
    vrs.set_tag("ml2.session_name",  session.name());
    vrs.set_tag("ml2.profile",       profile.name);
    vrs.set_tag("ml2.duration_s",    std::to_string(duration_s));
    vrs.set_tag("ml2.start_time_ns", std::to_string(start_time_ns));
    vrs.set_tag("ml2.streams.rgb",     std::to_string(enable_rgb));
    vrs.set_tag("ml2.streams.depth",   std::to_string(enable_depth));
    vrs.set_tag("ml2.streams.world",   std::to_string(enable_world));
    vrs.set_tag("ml2.streams.head",    std::to_string(enable_head));
    vrs.set_tag("ml2.streams.eye",     std::to_string(enable_eye));
    vrs.set_tag("ml2.streams.hand",    std::to_string(enable_hand));
    vrs.set_tag("ml2.streams.imu",     std::to_string(enable_imu));
    vrs.set_tag("ml2.streams.audio",   std::to_string(enable_audio));
    vrs.set_tag("ml2.streams.meshing", std::to_string(enable_meshing));

    if (!vrs.open(vrs_path)) {
        LOGE("Failed to open VRS file");
        return;
    }

    // ------------------------------------------------------------------
    // 3. Initialise captures (all receive the VrsWriter pointer)
    // ------------------------------------------------------------------

    ml2::RgbCapture rgb;
    if (enable_rgb) {
        ml2::RgbCaptureConfig cfg;
        cfg.vrs_writer    = &vrs;
        cfg.width         = profile.rgb_width;
        cfg.height        = profile.rgb_height;
        cfg.fps           = profile.rgb_fps;
        cfg.jpeg_quality  = profile.rgb_jpeg_quality;
        cfg.use_h264      = profile.rgb_use_h264;
        cfg.h264_bitrate  = profile.rgb_h264_bitrate;
        cfg.writer_threads = profile.rgb_writer_threads;
        cfg.queue_depth   = profile.rgb_queue_depth;
        if (!rgb.init(cfg)) {
            LOGE("RGB capture init failed");
            enable_rgb = false;
        }
    }

    ml2::PerceptionCapture perception;
    bool perception_inited = false;
    if (enable_head || enable_eye || enable_hand) {
        ml2::PerceptionCaptureConfig cfg;
        cfg.vrs_writer          = &vrs;
        cfg.poll_rate_hz        = profile.perception_poll_rate_hz;
        cfg.enable_head_pose    = enable_head;
        cfg.enable_eye_tracking = enable_eye;
        cfg.enable_hand_tracking = enable_hand;
        if (perception.init(cfg)) {
            perception_inited = true;
        } else {
            LOGE("Perception capture init failed");
            enable_head = enable_eye = enable_hand = false;
        }
    }

    // Wire the head-tracking handle into RGB so MLCVCameraGetFramePose can
    // query per-frame camera extrinsics. Harmless if either side is disabled.
    if (enable_rgb && perception_inited) {
        rgb.set_head_handle(perception.head_tracking_handle());
    }

    ml2::DepthCapture depth;
    if (enable_depth) {
        ml2::DepthCaptureConfig cfg;
        cfg.vrs_writer      = &vrs;
        cfg.use_short_range = profile.depth_short_range;
        if (!depth.init(cfg)) {
            LOGE("Depth capture init failed");
            enable_depth = false;
        }
    }

    ml2::WorldCameraCapture world_cams;
    if (enable_world) {
        ml2::WorldCameraCaptureConfig cfg;
        cfg.vrs_writer     = &vrs;
        cfg.enable_left    = profile.world_cam_left;
        cfg.enable_right   = profile.world_cam_right;
        cfg.enable_center  = profile.world_cam_center;
        cfg.jpeg_quality   = profile.world_cam_jpeg_quality;
        cfg.writer_threads = profile.world_cam_writer_threads;
        cfg.queue_depth    = profile.world_cam_queue_depth;
        if (!world_cams.init(cfg)) {
            LOGE("World camera init failed");
            enable_world = false;
        }
    }

    ml2::ImuCapture imu;
    if (enable_imu) {
        ml2::ImuCaptureConfig cfg;
        cfg.vrs_writer     = &vrs;
        cfg.sample_rate_us = profile.imu_sample_period_us;
        if (!imu.init(cfg)) {
            LOGE("IMU capture init failed");
            enable_imu = false;
        }
    }

    ml2::AudioCapture audio;
    if (enable_audio) {
        ml2::AudioCaptureConfig cfg;
        cfg.vrs_writer  = &vrs;
        cfg.channels    = profile.audio_channels;
        cfg.sample_rate = profile.audio_sample_rate;
        if (!audio.init(cfg)) {
            LOGE("Audio capture init failed");
            enable_audio = false;
        }
    }

    ml2::MeshingCapture meshing;
    if (enable_meshing) {
        ml2::MeshingCaptureConfig cfg;
        cfg.vrs_writer         = &vrs;
        cfg.poll_interval_s    = profile.meshing_poll_interval_s;
        cfg.bounds_extents_m   = profile.meshing_bounds_m;
        cfg.compute_normals    = profile.meshing_normals;
        cfg.compute_confidence = profile.meshing_confidence;
        cfg.lod                = profile.meshing_lod;
        if (!meshing.init(cfg)) {
            LOGE("Meshing capture init failed");
            enable_meshing = false;
        }
    }

    ml2::ThermalMonitor thermal;
    thermal.init(session.path(), app->activity->vm, app->activity->clazz);

    // ------------------------------------------------------------------
    // 4. Start captures
    // ------------------------------------------------------------------
    if (enable_rgb)         rgb.start();
    if (perception_inited)  perception.start();
    if (enable_depth)       depth.start();
    if (enable_world)       world_cams.start();
    if (enable_imu)         imu.start();
    if (enable_audio)       audio.start();
    if (enable_meshing)     meshing.start();

    LOGI("All sensors started — recording for %ds", duration_s);

    {
        char msg[128];
        if (duration_s >= 60)
            std::snprintf(msg, sizeof(msg), "Recording started (%dm %ds)",
                          duration_s / 60, duration_s % 60);
        else
            std::snprintf(msg, sizeof(msg), "Recording started (%ds)", duration_s);
        show_toast(app, msg);
        play_start_tone(app);
    }

    // ------------------------------------------------------------------
    // 5. Record loop
    // ------------------------------------------------------------------
    auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(duration_s);
    auto start_time  = std::chrono::steady_clock::now();
    auto last_status = start_time;

    auto pinch_start  = std::chrono::steady_clock::time_point{};
    bool pinch_active = false;
    bool gesture_stop = false;

    while (!app->destroyRequested &&
           std::chrono::steady_clock::now() < deadline &&
           !gesture_stop) {
        int events;
        android_poll_source* source;
        while (ALooper_pollOnce(100, nullptr, &events,
                                reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
            if (app->destroyRequested) break;
        }

        auto now = std::chrono::steady_clock::now();

        if (perception_inited && perception.is_both_hands_pinching(0.03f)) {
            if (!pinch_active) {
                pinch_active = true;
                pinch_start  = now;
            } else if (std::chrono::duration_cast<std::chrono::milliseconds>(
                           now - pinch_start).count() >= 2000) {
                LOGI("Two-hand pinch held 2s — stopping");
                show_toast(app, "Pinch detected — stopping...");
                gesture_stop = true;
            }
        } else {
            pinch_active = false;
        }

        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_status).count() >= 10) {
            last_status = now;
            auto elapsed   = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            auto remaining = duration_s - elapsed;
            int64_t storage_mb = ml2::StorageMonitor::available_mb(session.path());
            int thermal_status = thermal.sample();
            LOGI("Recording: %llds elapsed, %llds remaining | "
                 "rgb=%zu(%zu dropped) depth=%zu world=%zu/%zu/%zu(%zu dropped) "
                 "imu=%zu mesh=%u | storage=%lldMB thermal=%d",
                 (long long)elapsed, (long long)remaining,
                 rgb.frames_written(), rgb.frames_dropped(),
                 depth.frames_written(),
                 world_cams.frames_written(0), world_cams.frames_written(1),
                 world_cams.frames_written(2), world_cams.frames_dropped(),
                 imu.rows_written(), meshing.snapshots_written(),
                 (long long)storage_mb, thermal_status);

            if (ml2::StorageMonitor::is_low(session.path(), 1024)) {
                LOGE("Storage critically low (%lldMB)", (long long)storage_mb);
                show_toast(app, "Low storage — stopping!");
                break;
            }
            if (thermal.is_critical()) {
                LOGE("Thermal critical (status=%d)", thermal_status);
                show_toast(app, "Overheating — stopping!");
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Stop captures (reverse order)
    // ------------------------------------------------------------------
    play_stop_tone(app);
    show_toast(app, "Recording complete — saving data...");

    if (enable_meshing)     meshing.stop();
    if (enable_audio)       audio.stop();
    if (enable_imu)         imu.stop();
    if (enable_world)       world_cams.stop();
    if (enable_depth)       depth.stop();
    if (perception_inited)  perception.stop();
    if (enable_rgb)         rgb.stop();
    thermal.close();

    // ------------------------------------------------------------------
    // 7. Write final metadata tags and close VRS file
    // ------------------------------------------------------------------
    const int64_t end_time_ns = ml2::now_ns();
    vrs.set_tag("ml2.end_time_ns",     std::to_string(end_time_ns));
    vrs.set_tag("ml2.duration_actual_ns",
                std::to_string(end_time_ns - start_time_ns));
    vrs.set_tag("ml2.frames.rgb",     std::to_string(rgb.frames_written()));
    vrs.set_tag("ml2.frames.depth",   std::to_string(depth.frames_written()));
    vrs.set_tag("ml2.frames.wcam0",   std::to_string(world_cams.frames_written(0)));
    vrs.set_tag("ml2.frames.wcam1",   std::to_string(world_cams.frames_written(1)));
    vrs.set_tag("ml2.frames.wcam2",   std::to_string(world_cams.frames_written(2)));
    vrs.set_tag("ml2.rows.head_pose", std::to_string(perception.head_pose_rows()));
    vrs.set_tag("ml2.rows.eye",       std::to_string(perception.eye_tracking_rows()));
    vrs.set_tag("ml2.rows.hand",      std::to_string(perception.hand_tracking_rows()));
    vrs.set_tag("ml2.rows.imu",       std::to_string(imu.rows_written()));
    vrs.set_tag("ml2.bytes.audio",    std::to_string(audio.bytes_written()));
    vrs.set_tag("ml2.snapshots.mesh", std::to_string(meshing.snapshots_written()));
    vrs.set_tag("ml2.vertices.mesh",  std::to_string(meshing.total_vertices()));

    // Write calibration intrinsics as tags. Each camera captures fx/fy/cx/cy
    // and distortion from the first frame, so these are only known at session
    // end — hence the late set_tag calls (close() flushes them).
    auto write_intrinsics_tags = [&](const std::string& prefix,
                                      const ml2::MetadataWriter::Intrinsics& i,
                                      int width, int height) {
        vrs.set_tag("ml2.cal." + prefix + ".fx", std::to_string(i.fx));
        vrs.set_tag("ml2.cal." + prefix + ".fy", std::to_string(i.fy));
        vrs.set_tag("ml2.cal." + prefix + ".cx", std::to_string(i.cx));
        vrs.set_tag("ml2.cal." + prefix + ".cy", std::to_string(i.cy));
        if (width  > 0) vrs.set_tag("ml2.cal." + prefix + ".width",  std::to_string(width));
        if (height > 0) vrs.set_tag("ml2.cal." + prefix + ".height", std::to_string(height));
        // Space-separated distortion coefficients [k1, k2, p1, p2, k3].
        std::ostringstream ds;
        ds << i.distortion[0] << ' ' << i.distortion[1] << ' '
           << i.distortion[2] << ' ' << i.distortion[3] << ' ' << i.distortion[4];
        vrs.set_tag("ml2.cal." + prefix + ".distortion", ds.str());
    };

    if (enable_rgb && rgb.intrinsics().valid) {
        write_intrinsics_tags("rgb", rgb.intrinsics(),
                              profile.rgb_width, profile.rgb_height);
    }
    if (enable_depth && depth.intrinsics().valid) {
        write_intrinsics_tags("depth", depth.intrinsics(),
                              (int)depth.actual_width(), (int)depth.actual_height());
    }
    if (enable_world) {
        for (int c = 0; c < 3; ++c) {
            if (!world_cams.intrinsics(c).valid) continue;
            write_intrinsics_tags("wcam" + std::to_string(c),
                                   world_cams.intrinsics(c),
                                   (int)world_cams.actual_width(),
                                   (int)world_cams.actual_height());
        }
    }

    // Emit a second CONFIGURATION record per image stream with populated
    // intrinsics. VRS silently drops file tags added after createFileAsync,
    // so tags alone can't carry intrinsics captured from the first frame.
    // A second config record per stream works because VRS supports multiple
    // config records and the converter picks the last populated one.
    if (enable_rgb && rgb.intrinsics().valid) {
        const auto& i = rgb.intrinsics();
        ml2::VrsWriter::RgbConfig rc;
        rc.width   = profile.rgb_width;
        rc.height  = profile.rgb_height;
        rc.fps     = profile.rgb_fps;
        rc.bitrate = profile.rgb_use_h264 ? profile.rgb_h264_bitrate : 0;
        rc.fx = i.fx; rc.fy = i.fy; rc.cx = i.cx; rc.cy = i.cy;
        for (int k = 0; k < 5; ++k) rc.distortion[k] = i.distortion[k];
        vrs.write_rgb_config(rc);
    }
    if (enable_depth && depth.intrinsics().valid) {
        const auto& i = depth.intrinsics();
        ml2::VrsWriter::DepthConfig dc;
        dc.width  = (int)depth.actual_width();
        dc.height = (int)depth.actual_height();
        dc.fx = i.fx; dc.fy = i.fy; dc.cx = i.cx; dc.cy = i.cy;
        for (int k = 0; k < 5; ++k) dc.distortion[k] = i.distortion[k];
        vrs.write_depth_config(dc);
    }
    if (enable_world) {
        for (int c = 0; c < 3; ++c) {
            if (!world_cams.intrinsics(c).valid) continue;
            const auto& i = world_cams.intrinsics(c);
            ml2::VrsWriter::WorldCamConfig wc;
            wc.cam_index = c;
            wc.width  = (int)world_cams.actual_width();
            wc.height = (int)world_cams.actual_height();
            wc.fx = i.fx; wc.fy = i.fy; wc.cx = i.cx; wc.cy = i.cy;
            for (int k = 0; k < 5; ++k) wc.distortion[k] = i.distortion[k];
            vrs.write_world_cam_config(wc);
        }
    }

    vrs.close();  // Flush + write file index

    show_toast(app, ("Session saved! Pull: " + session.name() + ".vrs").c_str());

    LOGI("Recording complete:");
    LOGI("  profile:      %s", profile.name.c_str());
    LOGI("  vrs:          %s", vrs_path.c_str());
    LOGI("  rgb:          %zu frames (%zu dropped)",
         rgb.frames_written(), rgb.frames_dropped());
    LOGI("  depth:        %zu frames", depth.frames_written());
    LOGI("  world_cams:   %zu/%zu/%zu (%zu dropped)",
         world_cams.frames_written(0), world_cams.frames_written(1),
         world_cams.frames_written(2), world_cams.frames_dropped());
    LOGI("  head_pose:    %zu rows", perception.head_pose_rows());
    LOGI("  eye_tracking: %zu rows", perception.eye_tracking_rows());
    LOGI("  hand_tracking:%zu rows", perception.hand_tracking_rows());
    LOGI("  imu:          %zu rows", imu.rows_written());
    LOGI("  audio:        %zu bytes", audio.bytes_written());
    LOGI("  meshing:      %u snapshots, %llu vertices",
         meshing.snapshots_written(),
         (unsigned long long)meshing.total_vertices());

    while (!app->destroyRequested) {
        int events;
        android_poll_source* source;
        while (ALooper_pollOnce(200, nullptr, &events,
                                reinterpret_cast<void**>(&source)) >= 0) {
            if (source) source->process(app, source);
            if (app->destroyRequested) break;
        }
    }

    LOGI("ML2 Recorder exiting");
}
