#include "thermal_monitor.h"

#include <android/log.h>

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

bool ThermalMonitor::init(const std::string& session_dir,
                          JavaVM* vm, jobject activity) {
    vm_ = vm;
    activity_ = activity;

    if (!csv_.open(session_dir + "_thermal.csv",
                   {"timestamp_ns", "thermal_status"})) {
        LOGE("Failed to open thermal.csv");
        return false;
    }
    LOGI("Thermal monitor initialised (JNI)");
    return true;
}

int ThermalMonitor::sample() {
    int64_t ts = now_ns();
    int status = -1;

    if (vm_) {
        JNIEnv* env = nullptr;
        vm_->AttachCurrentThread(&env, nullptr);
        if (env) {
            // PowerManager pm = activity.getSystemService(Context.POWER_SERVICE);
            jclass context_class = env->FindClass("android/content/Context");
            jfieldID power_field = env->GetStaticFieldID(
                context_class, "POWER_SERVICE", "Ljava/lang/String;");
            jstring power_service =
                (jstring)env->GetStaticObjectField(context_class, power_field);

            jmethodID get_system_service = env->GetMethodID(
                context_class, "getSystemService",
                "(Ljava/lang/String;)Ljava/lang/Object;");
            jobject pm = env->CallObjectMethod(activity_, get_system_service,
                                               power_service);

            if (pm) {
                // int thermalStatus = pm.getCurrentThermalStatus();
                jclass pm_class = env->GetObjectClass(pm);
                jmethodID get_thermal = env->GetMethodID(
                    pm_class, "getCurrentThermalStatus", "()I");
                if (get_thermal) {
                    status = env->CallIntMethod(pm, get_thermal);
                }
                env->DeleteLocalRef(pm_class);
                env->DeleteLocalRef(pm);
            }

            env->DeleteLocalRef(power_service);
            env->DeleteLocalRef(context_class);
            vm_->DetachCurrentThread();
        }
    }

    last_status_ = status;
    csv_.write_row(ts, {std::to_string(status)});

    return status;
}

bool ThermalMonitor::is_critical(int threshold) {
    return last_status_ >= threshold;
}

void ThermalMonitor::close() {
    csv_.close();
}

} // namespace ml2
