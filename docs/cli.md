# CLI and batch

## `ofiqpy`

```bash
ofiqpy -i INPUT -o OUTPUT.csv
```

`INPUT` must exist and be a `.jpg`, `.jpeg`, `.png`, or `.bmp` file, or a directory
containing at least one such file. Directories are searched recursively. Input discovery,
config/model integrity verification, and model loading complete before the output file is
created. Invalid or empty input exits nonzero.

## `python -m ofiqpy.batch`

```bash
python -m ofiqpy.batch -i INPUT -o OUTPUT.csv [-w WORKERS] [--resume]
```

| Flag | Meaning |
|---|---|
| `-i` | Existing image or recursively searched directory |
| `-o` | Canonical assessment CSV |
| `-w` | Worker processes; default `1` |
| `--resume` | Skip exact full-path identities already in a valid output CSV |

Every worker loads a complete model graph. Increasing `-w` can multiply memory use and is
therefore an explicit operator choice. Multi-worker runs use Python's `spawn` process
context so a threaded OpenCV/ONNX runtime is never inherited through `fork`.

Resume validates the complete canonical header before reading identities. Duplicate rows
and truncated/widened rows in an existing output are rejected. Identity is the full
supplied/discovered path, not the basename, so two recursive captures named `capture.png`
remain distinct.

The first real assessment completes before the output file is opened. An unreadable image
or preprocessing failure raises `BatchAssessmentError` and makes the command exit nonzero;
no all-sentinel row is written for that item. If a later item fails, earlier flushed rows
remain valid and can be continued with `--resume`.

## CSV contract

The output is semicolon-delimited and uses the fixed canonical 28-component order:

```text
Filename;UnifiedQualityScore;...;NoHeadCoverings;UnifiedQualityScore.scalar;...;NoHeadCoverings.scalar;assessment_time_in_ms;
```

- One native/raw and one scalar column per component.
- Successful scalar values are in `[0, 100]`.
- `FailureToAssess` is encoded as raw `0.000000`, scalar `-1`.
- Filename fields use standard CSV quoting, including paths containing semicolons.
- The header retains OFIQSampleApp's terminal empty field; data rows do not add one.

The layout matches the canonical OFIQ profile. It is not a general adapter for C++
configurations that select a smaller active measure map.
