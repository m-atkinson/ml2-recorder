#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define ML2_HAS_SSE41 1
#else
#define ML2_HAS_SSE41 0
#endif

namespace ml2 {

/// Non-temporal streaming copy for reading from device-mapped (WC/UC) memory.
///
/// On x86_64 with SSE4.1, uses MOVNTDQA (non-temporal aligned loads) to read
/// from write-combining or uncacheable memory without polluting CPU caches,
/// and MOVNTPS (non-temporal stores) to write without read-for-ownership stalls.
///
/// This is 2-10x faster than regular memcpy for camera DMA buffers that are
/// mapped as uncacheable or write-combining.
///
/// Falls back to standard memcpy on non-x86 platforms.
inline void streaming_copy(void* __restrict dst, const void* __restrict src,
                           size_t bytes) {
#if ML2_HAS_SSE41
    auto* d = static_cast<uint8_t*>(dst);
    const auto* s = static_cast<const uint8_t*>(src);

    // Handle leading unaligned bytes (align source to 16 bytes).
    size_t src_misalign = reinterpret_cast<uintptr_t>(s) & 0xF;
    if (src_misalign && bytes >= 16) {
        size_t lead = 16 - src_misalign;
        if (lead > bytes) lead = bytes;
        std::memcpy(d, s, lead);
        d += lead;
        s += lead;
        bytes -= lead;
    }

    // Main loop: 64 bytes per iteration (4 × 128-bit SSE registers).
    // MOVNTDQA loads from WC/UC memory in full cache-line-sized chunks.
    // MOVNTPS stores bypass the cache (no read-for-ownership penalty).
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; ++i) {
        __m128i v0 = _mm_stream_load_si128(reinterpret_cast<const __m128i*>(s));
        __m128i v1 = _mm_stream_load_si128(reinterpret_cast<const __m128i*>(s + 16));
        __m128i v2 = _mm_stream_load_si128(reinterpret_cast<const __m128i*>(s + 32));
        __m128i v3 = _mm_stream_load_si128(reinterpret_cast<const __m128i*>(s + 48));

        _mm_stream_si128(reinterpret_cast<__m128i*>(d),      v0);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 16),  v1);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 32),  v2);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 48),  v3);

        s += 64;
        d += 64;
    }

    // Fence: ensure all non-temporal stores are globally visible.
    _mm_sfence();

    // Handle trailing bytes.
    size_t remaining = bytes - (chunks * 64);
    if (remaining > 0) {
        std::memcpy(d, s, remaining);
    }
#else
    std::memcpy(dst, src, bytes);
#endif
}

} // namespace ml2
