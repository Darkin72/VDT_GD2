# VDT-GD2: Đánh giá các phương pháp khử sương

Repository này dùng chung pipeline đánh giá cho GridDehazeNet, DCP và CAP trên
SOTS-Indoor, SOTS-Outdoor, O-HAZE và I-HAZE.

## Build HazeWaveNet trên Vast.ai bằng standalone BuildKit

### Phạm vi image

Image này chứa môi trường train/infer/evaluate của `solution.wavelet_dehaze`,
không phải môi trường tổng hợp cho mọi model trong repo. `main.py` chỉ in lời
chào; entrypoint train thật là `python -m solution.wavelet_dehaze.train`.
Module dùng relative imports nên không chạy trực tiếp file `train.py`.
Không có web server, port cần expose hay environment variable application bắt
buộc. `HW_TQDM` và `HW_TQDM_BATCH` chỉ điều khiển progress bar.

Repo dùng `uv` (`pyproject.toml`, `uv.lock`, Python >=3.12), nhưng dependency
root thiếu PyTorch/Pillow/numpy và các package evaluation. Container giữ
PyTorch/CUDA từ NGC và dùng `pip` với
`solution/wavelet_dehaze/requirements-container.txt` cho package còn lại;
không chạy `uv sync` để tránh thay thế bản PyTorch GPU trong base image.
Haar DWT được viết bằng PyTorch, không cần build extension hay cài OpenCV,
compiler hoặc system dependency bổ sung ngoài base cho phạm vi image này.

Dockerfile cũ được sửa ngay tại `solution/wavelet_dehaze/Dockerfile`: bỏ script
setup không tồn tại, bỏ UID/GID hard-code, cài dependency trước COPY source,
chỉ COPY HazeWaveNet và hai utility evaluation. CMD mặc định in `train --help`;
truyền command train khi chạy. Image chạy root mặc định; có thể dùng
`--user UID:GID` nếu thư mục output trên host được cấp quyền ghi tương ứng.
Không nhúng dataset, checkpoint, secret hay các model khác vào image.

Build yêu cầu Linux amd64, standalone `buildkitd` đang chạy và
`buildctl debug workers` thành công. Native snapshotter không cần cú pháp
Dockerfile đặc biệt. Không cần Docker daemon, systemd hoặc GPU khi build.
Ubuntu 22.04 của host không quyết định phiên bản Ubuntu bên trong image.
Base mặc định giữ nguyên `nvcr.io/nvidia/pytorch:26.04-py3` từ Dockerfile cũ;
cần xác nhận tag pull được và driver máy chạy tương thích với CUDA của base.
Có thể đổi bằng build arg `BASE_IMAGE`; không suy ra CUDA version từ H100.

### 1. Git clone trên Vast.ai

Những thay đổi containerization phải có trong revision được clone (workspace
này chưa được commit/push tự động). Một số dataset hiện được Git track;
`.dockerignore` chỉ loại chúng khỏi build context, không khỏi lịch sử Git.
Nếu Git server hỗ trợ partial clone, clone source cần thiết mà không checkout
dataset bằng các lệnh sau:

```bash
cd /root
git clone --filter=blob:none --no-checkout https://github.com/Darkin72/VDT_GD2.git
cd VDT_GD2
git sparse-checkout init --no-cone
git sparse-checkout set /README.md /.dockerignore /solution/wavelet_dehaze/ /utils/
git checkout
buildctl debug workers
```

DehazeDDPM, MB-TaylorFormerV2, UDPNet đang untracked khi chuẩn bị hướng dẫn;
GridDehazeNet là gitlink. Image này không phụ thuộc những thư mục đó.

### 2. Tải dataset trên Vast.ai

Không cần tải toàn bộ dataset về máy cá nhân. Tải vào host Vast.ai, ngoài repo
và ngoài image. Ví dụ ITS có slug được notebook Kaggle trong repo sử dụng:

```bash
python3 -m venv /root/kaggle-download-env
/root/kaggle-download-env/bin/python -m pip install kaggle
mkdir -p /root/clearair-data/ITS /root/clearair-output
/root/kaggle-download-env/bin/kaggle datasets download \
  -d balraj98/indoor-training-set-its-residestandard \
  -p /root/clearair-data/ITS --unzip
find /root/clearair-data/ITS -type d \( -name hazy -o -name clear \)
```

Nếu thiếu `venv`, cài `python3-venv` bằng apt trước bước này. Cần Internet,
quyền truy cập dataset và đủ disk cho archive lẫn dữ liệu giải nén. Nếu Kaggle
yêu cầu đăng nhập, cung cấp credential `/root/.kaggle/kaggle.json` và đặt
`chmod 600 /root/.kaggle/kaggle.json`; không đưa credential vào repo/image.
Với OTS, notebook dùng slug `brunobelloni/outdoor-training-set-ots-reside`.

Archive có thể lồng thư mục. Positional argument `data` phải là thư mục cha
trực tiếp chứa cả `hazy/` và `clear/`. Ví dụ nếu kết quả `find` là
`/root/clearair-data/ITS/ITS/hazy`, đường dẫn container cần truyền sẽ là
`/workspace/ClearAIR/dataset/ITS/ITS`, không phải `dataset/ITS`.
Không giả định tải Kaggle tự tạo train/val split. Với ITS chưa chia split,
dùng `--val-fraction 0.1`: trainer chia theo clear image để tránh leakage.
Với I-HAZE đã chia, truyền `I-HAZE/train` và `--val-data .../I-HAZE/val`.

```text
<data>/hazy/<ảnh sương>.png
<data>/clear/<ảnh rõ>.png
```

Trainer hỗ trợ tên giống nhau, hậu tố `_hazy` và biến thể tên RESIDE.
Train từ đầu không cần checkpoint. Inference chất lượng cần mount checkpoint
đã train và truyền `--checkpoint PATH`. Evaluate yêu cầu bốn checkpoint qua
`--checkpoint-i-haze`, `--checkpoint-o-hazy`, `--checkpoint-sots-its`,
`--checkpoint-sots-ots`; xem module `evaluate --help` để chọn đường dẫn.

**Dataset đã tải không tự đi theo image khi push/pull.** Máy chạy cuối cùng
phải tải dataset hoặc nhận bản copy rồi mount nó. Tải trước build theo thứ tự
này là để chuẩn bị dữ liệu, không phải để COPY dữ liệu vào image.

### 3. Build image bằng buildctl

Chạy từ root repo. Dockerfile nằm trong thư mục con nên phải truyền `filename`
tương đối với `--local dockerfile=.`; `.dockerignore` nằm ở root build context.

Xuất Docker archive, không cần Docker daemon:

```bash
cd /root/VDT_GD2
buildctl build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=. \
  --opt filename=solution/wavelet_dehaze/Dockerfile \
  --opt platform=linux/amd64 \
  --output type=docker,name=clearair-wavelet:latest,dest=/root/clearair-wavelet.tar \
  --progress=plain
```

Hoặc push trực tiếp lên Docker Hub (thay username):

```bash
buildctl build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=. \
  --opt filename=solution/wavelet_dehaze/Dockerfile \
  --opt platform=linux/amd64 \
  --output type=image,name=docker.io/YOUR_DOCKERHUB_USERNAME/clearair-wavelet:latest,push=true \
  --progress=plain
```

BuildKit dùng registry credential trong `$DOCKER_CONFIG/config.json`, mặc định
`~/.docker/config.json`. Không cần chạy `docker login`; tạo auth bằng prompt
để token không vào command history:

```bash
python3 - <<'PY'
import base64
import getpass
import json
import os
from pathlib import Path

config_dir = Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
username = input("Docker Hub username: ").strip()
token = getpass.getpass("Docker Hub access token: ")
auth = base64.b64encode(f"{username}:{token}".encode()).decode()
config.setdefault("auths", {})["https://index.docker.io/v1/"] = {"auth": auth}
config_path.write_text(json.dumps(config))
config_path.chmod(0o600)
PY
```

Auth là base64, không phải mã hóa; giữ file ngoài repo và không chia sẻ.
Nếu cấu hình có credential helper, helper đó phải có trên host. Nếu NGC yêu
cầu đăng nhập, thêm auth cho `nvcr.io` với username `$oauthtoken` và NGC API
key vào cùng cấu hình. Không hard-code credential trong Dockerfile/build arg.
Không cần SSH forwarding, private package credential hoặc build secret.

Để thay base đã được kiểm tra với driver, thêm option sau vào lệnh build:

```bash
--opt build-arg:BASE_IMAGE=nvcr.io/nvidia/pytorch:TAG_DA_KIEM_TRA
```

NGC image lớn; cần disk cho base, native snapshots và tar đầu ra. Dependency
được cache theo layer; native snapshotter có thể tốn disk và thời gian hơn.
Để tái lập chặt chẽ hơn, pin base bằng digest và khóa dependency sau khi build
thành công; requirements hiện dùng khoảng phiên bản, không phải lockfile.

### 4. Load và chạy trên máy có Docker/GPU

Đây là bước bên vận hành làm trên máy chạy cuối cùng, không phải yêu cầu
Docker daemon trên Vast.ai build host. Host cần NVIDIA driver tương thích và
GPU container runtime đã cấu hình. Không cài driver NVIDIA vào image.

```bash
docker load -i /path/to/clearair-wavelet.tar
docker run --rm --gpus all clearair-wavelet:latest \
  python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Nếu push Docker Hub, thay load bằng pull và dùng tên image đầy đủ.
Ví dụ train khi host có `/root/clearair-data/ITS/{hazy,clear}`:

```bash
mkdir -p /root/clearair-output
docker run --rm --gpus all --shm-size=8g \
  --mount type=bind,src=/root/clearair-data,dst=/workspace/ClearAIR/dataset,readonly \
  --mount type=bind,src=/root/clearair-output,dst=/workspace/ClearAIR/outputs \
  clearair-wavelet:latest \
  python -m solution.wavelet_dehaze.train /workspace/ClearAIR/dataset/ITS \
  --val-fraction 0.1 --device cuda --epochs 50 --size 256 \
  --batch-size 16 --micro-batch-size 4 --num-workers 4 \
  --pin-memory --persistent-workers --amp \
  --out /workspace/ClearAIR/outputs/haze_wavelet.pt \
  --history /workspace/ClearAIR/outputs/history.json
```

Điều chỉnh đường dẫn theo kết quả `find` bước 2. Batch/workers/shm là cấu hình
khởi đầu, không phải thông số benchmark H100 đã xác minh. Không cần `-p`,
environment secret, `--privileged` hoặc `--ipc=host`. Chỉ bật `--multi-gpu`
khi có ít nhất hai GPU; code hiện dùng DataParallel, không phải DDP.
Output phải mount writable để giữ checkpoint sau khi container bị xóa.
Trainer chỉ lưu weights/epoch, không có tùy chọn resume optimizer/scheduler.

### Giới hạn xác minh

Chỉ static checks tại workspace: source Python, COPY paths, dependency/CLI
flags, diff và shell syntax. Không chạy `docker build`, build/push image hay
train GPU tại đây. Cần xác minh ở Vast.ai: quyền pull/tag NGC, pip resolver,
quyền namespace BuildKit trong container thuê, Internet/disk, quyền Kaggle,
cấu trúc archive, driver GPU và quyền ghi mount tại máy chạy cuối cùng.

## Chạy trên Google Colab

### 1. Bật GPU và clone repository

Trong Colab chọn `Runtime` -> `Change runtime type` -> `T4 GPU` hoặc GPU mạnh hơn,
sau đó chạy:

```python
%cd /content
!git clone https://github.com/Darkin72/VDT_GD2.git
%cd /content/VDT_GD2
```

### 2. Cài dependency

```python
!pip install -q torch torchvision opencv-python tqdm numpy pillow scipy \
    scikit-image psutil openpyxl
```

Kiểm tra GPU:

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 3. Kiểm tra dataset

Đặt dataset tại `/content/VDT_GD2/dataset` theo cấu trúc:

```text
dataset/
├── I-HAZE/test/{clear,hazy}/
├── O-HAZY/test/{clear,hazy}/
└── Synthetic Objective Testing Set (SOTS) [RESIDE]/
    ├── indoor/test/{clear,hazy}/
    └── outdoor/test/{clear,hazy}/
```

Nếu dataset ở Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/dataset" /content/VDT_GD2/
```

### 4. Đánh giá GridDehazeNet

Script tự chọn checkpoint đúng cho từng nhóm:

- SOTS-Indoor và I-HAZE: `indoor_haze_best_3_6`.
- SOTS-Outdoor và O-HAZE: `outdoor_haze_best_3_6`.

```python
%cd /content/VDT_GD2
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --output-dir utils/evaluation_results/griddehazenet
```

Chạy thử một ảnh mỗi dataset trước:

```python
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --limit 1 \
    --output-dir utils/evaluation_results/griddehazenet_smoke
```

### 5. Đánh giá DCP và CAP

```python
!python utils/evaluate.py \
    --solution dcp \
    --dataset all \
    --split test \
    --output-dir utils/evaluation_results/dcp

!python utils/evaluate.py \
    --solution cap \
    --dataset all \
    --split test \
    --output-dir utils/evaluation_results/cap
```

### 6. File kết quả

Mỗi solution sinh ba file Excel:

```text
<solution>_dataset.xlsx  # SOTS-Indoor, SOTS-Outdoor, O-HAZE, I-HAZE
<solution>_domain.xlsx   # Real và Synthetic
<solution>_fog.xlsx      # Light, Medium và Heavy fog
```

Với GridDehazeNet, file tổng hợp nằm tại:

```text
utils/evaluation_results/griddehazenet/
├── griddehazenet_dataset.xlsx
├── griddehazenet_domain.xlsx
├── griddehazenet_fog.xlsx
├── summary_test.csv
└── performance_test.json
```

DCP và CAP lưu workbook trong thư mục run tương ứng:

```text
utils/evaluation_results/dcp/dcp/test/
utils/evaluation_results/cap/cap/test/
```

`performance_test.json` ghi hardware, CPU/RAM, GPU/VRAM nếu có, thời gian chạy
và FPS. Trong báo cáo nên phân biệt `FPS toàn bộ run` (bao gồm đọc ảnh, resize,
tính metric và ghi file) với `FPS suy luận trung bình` (chỉ thời gian model).
