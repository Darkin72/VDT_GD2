# VDT-GD2: Đánh giá các phương pháp khử sương

Repository này dùng chung pipeline đánh giá cho GridDehazeNet, DCP và CAP trên
SOTS-Indoor, SOTS-Outdoor, O-HAZE và I-HAZE.

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
