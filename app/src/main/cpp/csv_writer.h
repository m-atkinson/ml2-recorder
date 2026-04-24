#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace ml2 {

// Writes CSV files with a fixed schema. Thread-safe — multiple sensor threads
// can call write_row() concurrently. Flushes periodically for crash resilience.
class CsvWriter {
public:
    // Opens the file and writes the header row.
    // Returns false if the file cannot be opened.
    bool open(const std::string& filepath, const std::vector<std::string>& columns);

    // Writes a single row. Values must match the column count from open().
    // Thread-safe.
    void write_row(const std::vector<std::string>& values);

    // Convenience: write a row starting with a timestamp_ns, followed by string values.
    void write_row(int64_t timestamp_ns, const std::vector<std::string>& values);

    // Flush buffered data to disk.
    void flush();

    // Close the file. Also called by destructor.
    void close();

    // Sort all rows by the first column (timestamp_ns) and rewrite the file.
    // Call this before close() for multi-writer CSVs where rows may arrive
    // out of order.  No-op if the file has no path or zero rows.
    void sort_by_timestamp_and_close();

    // Number of data rows written (excludes header).
    size_t row_count() const;

    ~CsvWriter();

    // Non-copyable, movable
    CsvWriter() = default;
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    CsvWriter(CsvWriter&& other) noexcept;
    CsvWriter& operator=(CsvWriter&& other) noexcept;

private:
    std::string filepath_;
    std::string header_;
    std::ofstream file_;
    std::mutex mutex_;
    size_t column_count_ = 0;
    size_t row_count_ = 0;
    size_t rows_since_flush_ = 0;
    static constexpr size_t kFlushInterval = 100; // flush every N rows
};

} // namespace ml2
