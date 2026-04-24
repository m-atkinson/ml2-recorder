#include "csv_writer.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include <android/log.h>
#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

bool CsvWriter::open(const std::string& filepath,
                     const std::vector<std::string>& columns) {
    std::lock_guard<std::mutex> lock(mutex_);
    filepath_ = filepath;
    errno = 0;
    file_.open(filepath, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        LOGE("CsvWriter::open failed for '%s': errno=%d (%s)",
             filepath.c_str(), errno, std::strerror(errno));
        return false;
    }

    column_count_ = columns.size();
    row_count_ = 0;
    rows_since_flush_ = 0;

    // Build and save header for potential re-write during sort.
    std::ostringstream hdr;
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) hdr << ',';
        hdr << columns[i];
    }
    header_ = hdr.str();

    file_ << header_ << '\n';
    file_.flush();
    return true;
}

void CsvWriter::write_row(const std::vector<std::string>& values) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_.is_open()) return;

    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) file_ << ',';
        file_ << values[i];
    }
    file_ << '\n';
    ++row_count_;
    ++rows_since_flush_;

    if (rows_since_flush_ >= kFlushInterval) {
        file_.flush();
        rows_since_flush_ = 0;
    }
}

void CsvWriter::write_row(int64_t timestamp_ns,
                           const std::vector<std::string>& values) {
    std::vector<std::string> row;
    row.reserve(1 + values.size());
    row.push_back(std::to_string(timestamp_ns));
    row.insert(row.end(), values.begin(), values.end());
    write_row(row);
}

void CsvWriter::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        rows_since_flush_ = 0;
    }
}

void CsvWriter::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

size_t CsvWriter::row_count() const {
    return row_count_;
}

void CsvWriter::sort_by_timestamp_and_close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
            file_.close();
        }
    }

    if (filepath_.empty() || row_count_ == 0) return;

    // Read all data rows.
    std::ifstream in(filepath_);
    if (!in.is_open()) return;

    std::string header_line;
    std::getline(in, header_line);  // skip header

    std::vector<std::string> rows;
    rows.reserve(row_count_);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) rows.push_back(std::move(line));
    }
    in.close();

    // Sort by the first column (timestamp_ns) — numeric comparison.
    std::sort(rows.begin(), rows.end(),
        [](const std::string& a, const std::string& b) {
            // Parse timestamp from start of each line up to the first comma.
            auto ts = [](const std::string& s) -> int64_t {
                auto pos = s.find(',');
                return std::stoll(s.substr(0, pos));
            };
            return ts(a) < ts(b);
        });

    // Rewrite the file.
    std::ofstream out(filepath_, std::ios::out | std::ios::trunc);
    out << header_line << '\n';
    for (const auto& r : rows) {
        out << r << '\n';
    }
    out.flush();
    out.close();
}

CsvWriter::~CsvWriter() {
    // Don't lock in destructor — assume single owner at this point
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

CsvWriter::CsvWriter(CsvWriter&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    filepath_ = std::move(other.filepath_);
    header_ = std::move(other.header_);
    file_ = std::move(other.file_);
    column_count_ = other.column_count_;
    row_count_ = other.row_count_;
    rows_since_flush_ = other.rows_since_flush_;
    other.column_count_ = 0;
    other.row_count_ = 0;
    other.rows_since_flush_ = 0;
}

CsvWriter& CsvWriter::operator=(CsvWriter&& other) noexcept {
    if (this != &other) {
        close();
        std::lock_guard<std::mutex> lock(other.mutex_);
        filepath_ = std::move(other.filepath_);
        header_ = std::move(other.header_);
        file_ = std::move(other.file_);
        column_count_ = other.column_count_;
        row_count_ = other.row_count_;
        rows_since_flush_ = other.rows_since_flush_;
        other.column_count_ = 0;
        other.row_count_ = 0;
        other.rows_since_flush_ = 0;
    }
    return *this;
}

} // namespace ml2
