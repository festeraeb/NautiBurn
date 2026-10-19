# Dynamic MoE Prefetch Injector — CESARops Fleet

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Host (CPU)                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Prefetch      │  │ Check &      │  │ Background           │  │
│  │ Queue         │  │ Bind         │  │ Cleanup Thread       │  │
│  └──────┬────────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                  │                     │              │
│  ┌──────▼──────────────────▼─────────────────────▼───────────┐  │
│  │  LayerPrefetchRing (per layer, 256 layers)                 │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │  TransientSlot[16] per layer                         │   │  │
│  │  │  - expert_id, dev_ptr, event, state                  │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
              cudaStreamNonBlocking (DMA)
                          │
┌─────────────────────────────────────────────────────────────────┐
│  Device (GPU) — Pascal sm_60+                                   │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐   │
│  │ Tier-1: Pinned  │  │ Tier-2: Transient Prefetch Buffer    │   │
│  │ Frequent Experts│  │ (16 slots × layer × expert_size)     │   │
│  │ (≥60% VRAM)     │  │                                     │   │
│  └─────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Complete Implementation

```cpp
#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cassert>

// ============================================================================
// Configuration Constants
// ============================================================================

namespace cesarops {

static constexpr int kNumLayers       = 40;
static constexpr int kExpertsPerLayer = 256;
static constexpr int kTopK            = 8;
static constexpr int kTier2Slots      = 16;  // transient slots per layer
static constexpr int kTier1Ratio      = 60;  // percent of VRAM reserved for Tier-1

// ============================================================================
// Struct: TransientSlot
// ============================================================================

struct TransientSlot {
    std::atomic<int>    expert_id;   // -1 when free
    std::atomic<bool>   in_use;      // true while slot holds a pending/ready transfer
    cudaEvent_t         completion_event;
    void*               dev_ptr;     // device pointer into Tier-2 buffer
    cudaStream_t        dma_stream;  // stream on which the transfer was issued

    TransientSlot()
        : expert_id(-1)
        , in_use(false)
        , dev_ptr(nullptr)
    {
        cudaEventCreate(&completion_event,

[Reasoning Trace]:
Let me design a robust Dynamic MoE Prefetch Injector for the CESARops fleet. This is a sophisticated system that needs to handle:

1. Dual-tier VRAM management (Tier-1 pinned frequent experts, Tier-2 transient prefetch buffer)
2. Non-blocking async prefetching
3. Thread-safe ring buffer management per layer
4. Zero host synchronization blocking

Let me think through the architecture carefully:

**Key Design Decisions:**

1. **Thread Safety**: Use atomic operations for ring buffer head/tail pointers, and mutex for slot allocation. Since we're doing zero host sync, we need to be careful about host-side synchronization.


2. **Ring Buffer**: Each layer gets its own ring buffer with 16 slots, where each slot tracks an expert ID, a device pointer, and a completion event.
3. **Prefetch Flow**: When a request comes in, I'll scan the ring buffer for an available slot, issue an async memory copy from pinned host memory, and record a completion event on the DMA stream.
4. **Check and Bind**: This queries the ring buffer for a matching expert, checks if its event has completed, and returns the device pointer if ready.
5. **Fallback**: If the expert isn't in the ring buffer or the prefetch hasn't finished, I'll fall back to Tier-1 or trigger a synchronous copy.

Now I'm defining the core data structures: a TransientSlot to track each slot's state and completion event, a PrefetchRequest to hold the parameters for each prefetch operation, and a LayerPrefetchRing to manage the ring buffer per layer.

Defining the class structure... I'm laying out the core data structures now—each layer gets its own ring buffer with 16 slots, and I'm tracking pinned host memory pointers for the Tier-2 buffer along with a completion event for synchronization. The `issue_prefetch` method will scan for an available slot, record the prefetch operation on the DMA stream, and mark it as pending. For `check_and_bind_expert`, I'll query the ring buffer to see if the requested expert is ready, and if not, fall back to Tier-1 or trigger a synchronous copy. I'm also adding a background worker thread that periodically checks all ring buffers to clean up completed prefetches and free up slots.

Writing the implementation... I'm realizing the fallback logic needs to be more robust—if the expert isn't in the ring buffer, I should check Tier-1 first, and if that's not there either, fall back to a synchronous copy. I also need to handle the case where the prefetch is still in progress by returning false, and I should add a background thread to periodically clean up completed prefetches and free up slots. Let me refine the implementation with proper thread safety, fallback handling, and a cleaner overall structure.
