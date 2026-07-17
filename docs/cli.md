# CLI & Batch

## `ofiqpy` — single / small runs

Mirrors `OFIQSampleApp`'s interface:

```bash
ofiqpy -i <image|directory> -o <out.csv>
```

- `-i` accepts a single image file or a directory (searched recursively for
  `.jpg/.jpeg/.png/.bmp`).
- `-o` writes an OFIQ-format CSV.

## `python -m ofiqpy.batch` — parallel batch

For large directories, the batch runner processes images across worker processes and is
resumable:

```bash
python -m ofiqpy.batch -i <directory> -o <out.csv> [-w N] [--resume]
```

| Flag | Meaning |
|---|---|
| `-i` | input directory (recursive) or file |
| `-o` | output CSV |
| `-w` | worker processes (default: number of CPUs) |
| `--resume` | skip images already present in the output CSV |

Each worker builds its own pipeline (ONNX / `cv2.ml` models are not shared across
processes). Rows are flushed as they complete, so an interrupted run can be resumed with
`--resume`.

## Output format

Semicolon-delimited, matching `OFIQSampleApp`:

```
Filename;UnifiedQualityScore;BackgroundUniformity;...;<component>;...;
        UnifiedQualityScore.scalar;...;<component>.scalar;...;assessment_time_in_ms
```

- One raw (native) column and one `.scalar` (0–100) column per component, in OFIQ's order.
- A component that fails to assess emits the OFIQ sentinel: raw `0`, scalar `-1`.
- Load with pandas:

```python
import pandas as pd
df = pd.read_csv("out.csv", sep=";")
df["UnifiedQualityScore.scalar"].describe()
```
