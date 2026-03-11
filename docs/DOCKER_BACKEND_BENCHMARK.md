# Docker Backend Benchmark

This project can run both `onnx` and `trt` inside the same Docker stack.

## Start ONNX Locally in Docker

PowerShell:

```powershell
$env:ANIMATION_BACKEND = "onnx"
docker compose up --build -d
```

Health check:

```powershell
$headers = @{ Authorization = "Bearer change-me" }
Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/health" -Headers $headers
```

Stop the stack:

```powershell
docker compose down
```

## Benchmark ONNX vs TRT in Docker

The benchmark script starts the Docker stack twice, once with `onnx` and once with `trt`,
submits the same audio job to both backends, waits for completion, and compares:

- API acknowledgement time
- queue wait time
- worker runtime
- runner elapsed time from `run_report.json`
- total job wall time

### Example

PowerShell:

```powershell
python .\scripts\benchmark_docker_backends.py `
  --audio C:\path\to\voice.mp3 `
  --source-image C:\path\to\person.png `
  --build
```

If the token is already defined in `.env`, the script reuses it automatically.
Otherwise pass it explicitly:

```powershell
python .\scripts\benchmark_docker_backends.py `
  --audio C:\path\to\voice.mp3 `
  --source-image C:\path\to\person.png `
  --token change-me `
  --build
```

### Output

The script prints a compact console table and writes a JSON summary to:

```text
output_fasterliveportrait/backend_benchmark/docker_backend_benchmark_<timestamp>.json
```

### Notes

- `--source-image` is the safest option for reproducible benchmarks because it avoids depending on one local frame path.
- If you omit `--source-image`, `--source-frame` must be relative to the project root.
- The script waits for health and warmup completion before submitting the benchmark job.
- Use `--keep-last-container` if you want the last backend container to stay up after the benchmark.
