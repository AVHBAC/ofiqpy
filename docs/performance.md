# Runtime performance

## Decision

Batch execution defaults to one worker. On the reviewed host, two and four workers were
slower than one worker for both the 0.1.1 baseline and the 0.2.0 candidate while each
additional worker loaded another complete model graph. Explicit `--workers` remains
available for operators to benchmark on their own hardware.

This is a host-specific operational result, not a cross-platform performance guarantee.
The redistribution-safe machine record is
[`runtime-benchmark-20260815.json`](evidence/runtime-benchmark-20260815.json).

## Method

The benchmark compared baseline commit
`c80fb388fe1ccecd17f60f296a1ef31bd95a1ee3` with the reviewed 0.2.0 Python source tree.
It used all 28 real BSI conformance PNGs for single-process latency and a fixed 64-image
subsample of the licensed, locally installed CelebA original-wild JPEGs for batch scaling.
No image or per-image identity is redistributed.

Each scenario received one warm-up followed by three measured repetitions. Baseline and
candidate order alternated by repetition. Every case required no concurrent OFIQ process,
at least 64 GiB available memory, and no more than 8 percent host CPU over two seconds
before launch. Peak RSS is the complete process tree sampled every 20 ms. The host had 24
physical / 32 logical x86-64 CPUs, 134,747,688,960 bytes of memory, and Python 3.11.14.

All six measured single-process runs produced the same scalar digest. Every batch case
wrote exactly 64 unique rows; both variants consistently identified the same one real
image as all-component `FailureToAssess`.

## Single-process medians

| Metric | 0.1.1 baseline | 0.2.0 candidate | Candidate change |
|---|---:|---:|---:|
| Cold first assessment | 2.601 s | 2.693 s | +3.6% |
| Warm throughput | 2.004 images/s | 2.060 images/s | +2.8% |
| Warm median latency | 495.7 ms | 483.3 ms | -2.5% |
| Process wall time | 17.130 s | 17.062 s | -0.4% |
| Peak process-tree RSS | 1.467 GiB | 1.211 GiB | -17.5% |

The small cold-start regression is outweighed operationally by lower warm latency and a
275,165,184-byte reduction in median peak RSS, but applications dominated by one-shot
assessment should measure their own initialization path.

## Batch medians on 64 real images

| Variant | Workers | Throughput | Elapsed | Peak process-tree RSS |
|---|---:|---:|---:|---:|
| 0.1.1 baseline | 1 | 2.656 images/s | 24.101 s | 1.446 GiB |
| 0.1.1 baseline | 2 | 2.557 images/s | 25.026 s | 2.778 GiB |
| 0.1.1 baseline | 4 | 2.242 images/s | 28.551 s | 5.094 GiB |
| 0.2.0 candidate | 1 | 2.649 images/s | 24.162 s | 1.292 GiB |
| 0.2.0 candidate | 2 | 2.391 images/s | 26.765 s | 2.450 GiB |
| 0.2.0 candidate | 4 | 2.140 images/s | 29.907 s | 4.723 GiB |

Relative to one worker, candidate throughput fell 9.7 percent at two workers and 19.2
percent at four workers. The result confirms one worker as the safe default for this
profile and host. It does not prohibit explicit concurrency where a different processor,
runtime provider, or workload demonstrates a measured benefit within its memory budget.
