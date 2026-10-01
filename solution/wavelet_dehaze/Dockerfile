ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/ClearAIR

COPY solution/wavelet_dehaze/requirements-container.txt /tmp/requirements-container.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-container.txt \
    && python -m pip check

COPY solution/wavelet_dehaze/ /workspace/ClearAIR/solution/wavelet_dehaze/
RUN mkdir -p /workspace/ClearAIR/utils
COPY utils/excel_report.py utils/evaluation_hardware.py /workspace/ClearAIR/utils/

RUN mkdir -p /workspace/ClearAIR/dataset /workspace/ClearAIR/outputs \
    && python -m solution.wavelet_dehaze.train --help \
    && python -m solution.wavelet_dehaze.infer --help \
    && python -m solution.wavelet_dehaze.evaluate --help

CMD ["python", "-m", "solution.wavelet_dehaze.train", "--help"]
