# Fetches the PaddleOCR model weights as plain files over HTTP — no paddle/paddlex import, so
# it can't repeat the "init crashes in some build daemons" issue that ruled out a RUN warmup
# below (that came from constructing the actual inference pipeline during build).
FROM python:3.11-slim AS model_downloader
RUN pip install --no-cache-dir "huggingface_hub[cli]>=0.28,<1"
# Must match the models thai_id.py's _get_ocr() pins: PP-OCRv5_mobile_det/th_PP-OCRv5_mobile_rec
# explicitly, plus PP-LCNet_x1_0_textline_ori for use_textline_orientation=True. Destination path
# matches paddlex's own default cache layout (~/.paddlex/official_models/<model>), so PaddleOCR()
# finds them already present at runtime instead of downloading.
RUN for model in PP-LCNet_x1_0_textline_ori PP-OCRv5_mobile_det th_PP-OCRv5_mobile_rec; do \
        hf download "PaddlePaddle/${model}" --local-dir "/root/.paddlex/official_models/${model}"; \
    done

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ccache \
        tesseract-ocr \
        tesseract-ocr-eng \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# fastmrz needs the custom MRZ-trained Tesseract model from tesseractMRZ.
# Resolve tessdata dir from the installed eng.traineddata so this survives
# Debian/tesseract version bumps.
# -f makes curl FAIL on HTTP errors (e.g. GitHub 429) instead of writing the error page to the
# file; --retry rides out transient rate-limits; the >1MB size gate rejects any junk that slips
# through (the real model is ~11MB — a 199-byte "429 Too Many Requests" page must not pass).
RUN TESSDATA_DIR="$(dirname "$(find /usr/share -name eng.traineddata | head -n1)")" \
    && curl -fSL --retry 5 --retry-all-errors --retry-delay 5 \
        -o "${TESSDATA_DIR}/mrz.traineddata" \
        https://github.com/DoubangoTelecom/tesseractMRZ/raw/master/tessdata_best/mrz.traineddata \
    && [ "$(wc -c < "${TESSDATA_DIR}/mrz.traineddata")" -gt 1000000 ]

COPY requirements.txt .
# paddlex[ocr] brings opencv-contrib-python==4.10.0.84 and checks for that exact package name at runtime
# (importlib.metadata), so we keep it instead of substituting the headless variant.
RUN pip install --no-cache-dir -r requirements.txt
# PaddleOCR: thread caps reduce allocator issues.
# Model weights are baked in by the model_downloader stage below (COPY, not a RUN warmup —
# constructing PaddleOCR() during build crashed some build daemons). Skip the runtime
# connectivity check to the model hosters since the files are already on disk.
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
# MKL-DNN (oneDNN) conv kernels: ~20-27% faster per scan (measured: 10 consecutive predict()
# calls on the mobile det/rec models, 32.3s -> 23.6s avg) with byte-identical output. Previously
# disabled over segfault worries in slim containers; paddlepaddle is already pinned to 3.2.x for
# an unrelated oneDNN/PIR regression in 3.3.x (see requirements.txt) — on 3.2.x with these mobile
# models, MKL-DNN ran crash-free across repeated trials, so there's no reason left to pay for it.
ENV FLAGS_use_mkldnn=1
# paddlex's own predictor thread pool ignores the OMP/OPENBLAS/MKL caps above and defaults to 10
# threads, oversubscribing past the 2-core CPU limit set in docker-compose.yml/k8s (contention,
# not real parallelism). Capping it to the actual core count measured 2x faster real /scan
# requests (106s -> 53s on the same image) with byte-identical output — same model, same math,
# just no thread thrashing. Bump this if the CPU limit elsewhere is ever raised past 2.
ENV PADDLE_PDX_CPU_NUM_THREADS=2
# fastmrz/pytesseract shell out to the `tesseract` binary, which ignores OMP_NUM_THREADS and
# spawns its own OpenMP thread pool sized to the node's real core count (measured: 4 threads on
# a 4-vCPU node, confirmed via a bare `tesseract` invocation vs `OMP_THREAD_LIMIT`), oversubscribing
# past the same CPU limit as above. OMP_THREAD_LIMIT is the var tesseract's OpenMP runtime actually
# honors — confirmed it drops the subprocess to 1 thread. Keep in sync with PADDLE_PDX_CPU_NUM_THREADS.
ENV OMP_THREAD_LIMIT=2

COPY --from=model_downloader /root/.paddlex/official_models /root/.paddlex/official_models
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
