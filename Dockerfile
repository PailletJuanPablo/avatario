FROM shaoguo/faster_liveportrait:v3

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    ANIMATION_BACKEND=trt \
    ANIMATION_TRT_RUNTIME=local \
    ANIMATION_TRT_PRECISION=fp16 \
    ANIMATION_WARMUP_ENABLED=1 \
    ANIMATION_AUDIO_MOTION_STRIDE=1 \
    LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/lib

# Runtime dependencies installed once at image build time.
RUN /root/miniconda3/bin/pip install --no-cache-dir \
    colorama \
    transformers==4.40.2 \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

COPY index.html /app/index.html
COPY realtime_stream_api.py /app/realtime_stream_api.py
COPY faster_liveportrait_runner.py /app/faster_liveportrait_runner.py
COPY faster_liveportrait_audio_to_pkl.py /app/faster_liveportrait_audio_to_pkl.py
COPY third_party/FasterLivePortrait /app/third_party/FasterLivePortrait

EXPOSE 8010

CMD ["/root/miniconda3/bin/python", "realtime_stream_api.py", "--host", "0.0.0.0", "--port", "8010", "--backend", "trt", "--trt-runtime", "local", "--trt-precision", "fp16"]
