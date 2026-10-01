# VDT-GD2: ÄÃ¡nh giÃ¡ cÃ¡c phÆ°Æ¡ng phÃ¡p khá»­ sÆ°Æ¡ng

Repository nÃ y dÃ¹ng chung pipeline Ä‘Ã¡nh giÃ¡ cho GridDehazeNet, DCP vÃ  CAP trÃªn
SOTS-Indoor, SOTS-Outdoor, O-HAZE vÃ  I-HAZE.

## Build HazeWaveNet trÃªn Vast.ai báº±ng standalone BuildKit

### Pháº¡m vi image

Image nÃ y chá»©a mÃ´i trÆ°á»ng train/infer/evaluate cá»§a `solution.wavelet_dehaze`,
khÃ´ng pháº£i mÃ´i trÆ°á»ng tá»•ng há»£p cho má»i model trong repo. `main.py` chá»‰ in lá»i
chÃ o; entrypoint train tháº­t lÃ  `python -m solution.wavelet_dehaze.train`.
Module dÃ¹ng relative imports nÃªn khÃ´ng cháº¡y trá»±c tiáº¿p file `train.py`.
KhÃ´ng cÃ³ web server, port cáº§n expose hay environment variable application báº¯t
buá»™c. `HW_TQDM` vÃ  `HW_TQDM_BATCH` chá»‰ Ä‘iá»u khiá»ƒn progress bar.

Repo dÃ¹ng `uv` (`pyproject.toml`, `uv.lock`, Python >=3.12), nhÆ°ng dependency
root thiáº¿u PyTorch/Pillow/numpy vÃ  cÃ¡c package evaluation. Container giá»¯
PyTorch/CUDA tá»« NGC vÃ  dÃ¹ng `pip` vá»›i
`solution/wavelet_dehaze/requirements-container.txt` cho package cÃ²n láº¡i;
khÃ´ng cháº¡y `uv sync` Ä‘á»ƒ trÃ¡nh thay tháº¿ báº£n PyTorch GPU trong base image.
Haar DWT Ä‘Æ°á»£c viáº¿t báº±ng PyTorch, khÃ´ng cáº§n build extension hay cÃ i OpenCV,
compiler hoáº·c system dependency bá»• sung ngoÃ i base cho pháº¡m vi image nÃ y.

Dockerfile cÅ© Ä‘Æ°á»£c sá»­a ngay táº¡i `solution/wavelet_dehaze/Dockerfile`: bá» script
setup khÃ´ng tá»“n táº¡i, bá» UID/GID hard-code, cÃ i dependency trÆ°á»›c COPY source,
chá»‰ COPY HazeWaveNet vÃ  hai utility evaluation. CMD máº·c Ä‘á»‹nh in `train --help`;
truyá»n command train khi cháº¡y. Image cháº¡y root máº·c Ä‘á»‹nh; cÃ³ thá»ƒ dÃ¹ng
`--user UID:GID` náº¿u thÆ° má»¥c output trÃªn host Ä‘Æ°á»£c cáº¥p quyá»n ghi tÆ°Æ¡ng á»©ng.
KhÃ´ng nhÃºng dataset, checkpoint, secret hay cÃ¡c model khÃ¡c vÃ o image.

Build yÃªu cáº§u Linux amd64, standalone `buildkitd` Ä‘ang cháº¡y vÃ 
`buildctl debug workers` thÃ nh cÃ´ng. Native snapshotter khÃ´ng cáº§n cÃº phÃ¡p
Dockerfile Ä‘áº·c biá»‡t. KhÃ´ng cáº§n Docker daemon, systemd hoáº·c GPU khi build.
Ubuntu 22.04 cá»§a host khÃ´ng quyáº¿t Ä‘á»‹nh phiÃªn báº£n Ubuntu bÃªn trong image.
Base máº·c Ä‘á»‹nh giá»¯ nguyÃªn `nvcr.io/nvidia/pytorch:26.04-py3` tá»« Dockerfile cÅ©;
cáº§n xÃ¡c nháº­n tag pull Ä‘Æ°á»£c vÃ  driver mÃ¡y cháº¡y tÆ°Æ¡ng thÃ­ch vá»›i CUDA cá»§a base.
CÃ³ thá»ƒ Ä‘á»•i báº±ng build arg `BASE_IMAGE`; khÃ´ng suy ra CUDA version tá»« H100.

### 1. Git clone trÃªn Vast.ai

Nhá»¯ng thay Ä‘á»•i containerization pháº£i cÃ³ trong revision Ä‘Æ°á»£c clone (workspace
nÃ y chÆ°a Ä‘Æ°á»£c commit/push tá»± Ä‘á»™ng). Má»™t sá»‘ dataset hiá»‡n Ä‘Æ°á»£c Git track;
`.dockerignore` chá»‰ loáº¡i chÃºng khá»i build context, khÃ´ng khá»i lá»‹ch sá»­ Git.
Náº¿u Git server há»— trá»£ partial clone, clone source cáº§n thiáº¿t mÃ  khÃ´ng checkout
dataset báº±ng cÃ¡c lá»‡nh sau:

```bash
cd /root
git clone --filter=blob:none --no-checkout https://github.com/Darkin72/VDT_GD2.git
cd VDT_GD2
git sparse-checkout init --no-cone
git sparse-checkout set /README.md /.dockerignore /scripts/ /solution/wavelet_dehaze/ /utils/
git checkout
buildctl debug workers
```

DehazeDDPM, MB-TaylorFormerV2, UDPNet Ä‘ang untracked khi chuáº©n bá»‹ hÆ°á»›ng dáº«n;
GridDehazeNet lÃ  gitlink. Image nÃ y khÃ´ng phá»¥ thuá»™c nhá»¯ng thÆ° má»¥c Ä‘Ã³.

### 2. Táº£i dataset trÃªn Vast.ai

KhÃ´ng cáº§n táº£i toÃ n bá»™ dataset vá» mÃ¡y cÃ¡ nhÃ¢n. Táº£i vÃ o host Vast.ai, ngoÃ i repo
vÃ  ngoÃ i image. Táº£i cáº£ ITS, OTS vÃ  SOTS test theo notebook Colab:

```bash
python3 -m venv /root/kaggle-download-env
/root/kaggle-download-env/bin/python -m pip install --upgrade kaggle
mkdir -p /root/clearair-output
chmod +x scripts/download_kaggle_datasets.sh
bash scripts/download_kaggle_datasets.sh
```

Náº¿u thiáº¿u `venv`, cÃ i `python3-venv` báº±ng apt trÆ°á»›c bÆ°á»›c nÃ y. Cáº§n Internet,
quyá»n truy cáº­p dataset vÃ  Ä‘á»§ disk cho archive láº«n dá»¯ liá»‡u giáº£i nÃ©n. Náº¿u Kaggle
yÃªu cáº§u Ä‘Äƒng nháº­p, cung cáº¥p credential `/root/.kaggle/kaggle.json` vÃ  Ä‘áº·t
`chmod 600 /root/.kaggle/kaggle.json`; khÃ´ng Ä‘Æ°a credential vÃ o repo/image.
KhÃ´ng dÃ¹ng `datasets list --max-size 5`: giá»›i háº¡n Ä‘Ã³ tÃ­nh theo bytes.

Archive cÃ³ thá»ƒ lá»“ng thÆ° má»¥c. Positional argument `data` pháº£i lÃ  thÆ° má»¥c cha
trá»±c tiáº¿p chá»©a cáº£ `hazy/` vÃ  `clear/`. VÃ­ dá»¥ náº¿u káº¿t quáº£ `find` lÃ 
`/root/clearair-data/ITS/ITS/hazy`, Ä‘Æ°á»ng dáº«n container cáº§n truyá»n sáº½ lÃ 
`/workspace/ClearAIR/dataset/ITS/ITS`, khÃ´ng pháº£i `dataset/ITS`.
KhÃ´ng giáº£ Ä‘á»‹nh táº£i Kaggle tá»± táº¡o train/val split. Vá»›i ITS chÆ°a chia split,
dÃ¹ng `--val-fraction 0.1`: trainer chia theo clear image Ä‘á»ƒ trÃ¡nh leakage.
Vá»›i I-HAZE Ä‘Ã£ chia, truyá»n `I-HAZE/train` vÃ  `--val-data .../I-HAZE/val`.

```text
<data>/hazy/<áº£nh sÆ°Æ¡ng>.png
<data>/clear/<áº£nh rÃµ>.png
```

Trainer há»— trá»£ tÃªn giá»‘ng nhau, háº­u tá»‘ `_hazy` vÃ  biáº¿n thá»ƒ tÃªn RESIDE.
Train tá»« Ä‘áº§u khÃ´ng cáº§n checkpoint. Inference cháº¥t lÆ°á»£ng cáº§n mount checkpoint
Ä‘Ã£ train vÃ  truyá»n `--checkpoint PATH`. Evaluate nháº­n checkpoint qua
`--checkpoint-i-haze`, `--checkpoint-o-hazy`, `--checkpoint-sots-its`,
`--checkpoint-sots-ots`; má»™t file cÃ³ thá»ƒ dÃ¹ng cho nhiá»u bá»™ test.

Äá»ƒ cÃ³ Ä‘á»§ 4 XLSX, cáº§n test trÃªn SOTS-Indoor, SOTS-Outdoor, I-HAZE vÃ  O-HAZY,
khÃ´ng chá»‰ train ITS/OTS. I-HAZE/O-HAZY test Ä‘Ã£ cÃ³ trong repo; sparse clone
bÆ°á»›c 1 chÆ°a checkout chÃºng. Láº¥y riÃªng hai bá»™ test trÃªn Vast.ai:

```bash
cd /root/VDT_GD2
git sparse-checkout add /dataset/I-HAZE/test/ /dataset/O-HAZY/test/
cp -a dataset/I-HAZE /root/clearair-data/
cp -a dataset/O-HAZY /root/clearair-data/
```

Äáº·t SOTS Ä‘Ã£ giáº£i nÃ©n vÃ o cáº¥u trÃºc dÆ°á»›i Ä‘Ã¢y. Náº¿u archive chá»‰ cÃ³
`indoor/{clear,hazy}` vÃ  `outdoor/{clear,hazy}`, táº¡o symlink tÆ°Æ¡ng Ä‘á»‘i nhÆ°
notebook. Thay `SOTS_SOURCE` báº±ng thÆ° má»¥c chá»©a trá»±c tiáº¿p `indoor/` vÃ  `outdoor/`:

```bash
SOTS_SOURCE=/root/clearair-data/SOTS-download
SOTS_TARGET='/root/clearair-data/Synthetic Objective Testing Set (SOTS) [RESIDE]'
for domain in indoor outdoor; do
  mkdir -p "$SOTS_TARGET/$domain/test"
  for kind in clear hazy; do
    source="$SOTS_SOURCE/$domain/$kind"
    test -d "$source" || { echo "Missing: $source"; exit 1; }
    target="$SOTS_TARGET/$domain/test/$kind"
    ln -s "$(realpath --relative-to="$(dirname "$target")" "$source")" "$target"
  done
done
```

Symlink pháº£i trá» bÃªn trong `/root/clearair-data` Ä‘á»ƒ váº«n dÃ¹ng Ä‘Æ°á»£c khi mount.
Cáº¥u trÃºc test cuá»‘i cÃ¹ng:

```text
/root/clearair-data/I-HAZE/test/{clear,hazy}/
/root/clearair-data/O-HAZY/test/{clear,hazy}/
/root/clearair-data/Synthetic Objective Testing Set (SOTS) [RESIDE]/indoor/test/{clear,hazy}/
/root/clearair-data/Synthetic Objective Testing Set (SOTS) [RESIDE]/outdoor/test/{clear,hazy}/
```

**Dataset Ä‘Ã£ táº£i khÃ´ng tá»± Ä‘i theo image khi push/pull.** MÃ¡y cháº¡y cuá»‘i cÃ¹ng
pháº£i táº£i dataset hoáº·c nháº­n báº£n copy rá»“i mount nÃ³. Táº£i trÆ°á»›c build theo thá»© tá»±
nÃ y lÃ  Ä‘á»ƒ chuáº©n bá»‹ dá»¯ liá»‡u, khÃ´ng pháº£i Ä‘á»ƒ COPY dá»¯ liá»‡u vÃ o image.

### 3. Build image báº±ng buildctl

Cháº¡y tá»« root repo. Dockerfile náº±m trong thÆ° má»¥c con nÃªn pháº£i truyá»n `filename`
tÆ°Æ¡ng Ä‘á»‘i vá»›i `--local dockerfile=.`; `.dockerignore` náº±m á»Ÿ root build context.

Xuáº¥t Docker archive, khÃ´ng cáº§n Docker daemon:

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

#### ÄÄƒng nháº­p Docker Hub cho standalone BuildKit

TrÆ°á»›c khi push, táº¡o access token Docker Hub cÃ³ quyá»n ghi repository vÃ  cháº¡y
script Ä‘Äƒng nháº­p bÃªn dÆ°á»›i. ÄÃ¢y lÃ  auth cho BuildKit, khÃ´ng cáº§n Docker daemon.
Token khÃ´ng Ä‘Æ°á»£c ghi vÃ o source, Dockerfile hay history cá»§a shell.

Hoáº·c push trá»±c tiáº¿p lÃªn Docker Hub sau khi Ä‘Äƒng nháº­p (thay username):

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

BuildKit dÃ¹ng registry credential trong `$DOCKER_CONFIG/config.json`, máº·c Ä‘á»‹nh
`~/.docker/config.json`. KhÃ´ng cáº§n cháº¡y `docker login`; táº¡o auth báº±ng prompt
Ä‘á»ƒ token khÃ´ng vÃ o command history:

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
with open("/dev/tty") as terminal:
    print("Docker Hub username: ", end="", flush=True)
    username = terminal.readline().strip()
    token = getpass.getpass("Docker Hub access token: ", stream=terminal)
if not username or not token:
    raise ValueError("Username and token must not be empty")
auth = base64.b64encode(f"{username}:{token}".encode()).decode()
config.setdefault("auths", {})["https://index.docker.io/v1/"] = {"auth": auth}
config_path.write_text(json.dumps(config))
config_path.chmod(0o600)
PY
```

Script Ä‘á»c tá»« `/dev/tty` vÃ¬ stdin Ä‘ang chá»©a Python heredoc. TrÃªn mÃ¡y cÃ³ Docker
CLI cÃ³ thá»ƒ dÃ¹ng `docker login --username YOUR_DOCKERHUB_USERNAME`, nhÆ°ng
khÃ´ng cáº§n lá»‡nh Ä‘Ã³ trÃªn Vast.ai. Docker CLI login cÅ©ng khÃ´ng cáº§n daemon.

Auth lÃ  base64, khÃ´ng pháº£i mÃ£ hÃ³a; giá»¯ file ngoÃ i repo vÃ  khÃ´ng chia sáº».
Náº¿u cáº¥u hÃ¬nh cÃ³ credential helper, helper Ä‘Ã³ pháº£i cÃ³ trÃªn host. Náº¿u NGC yÃªu
cáº§u Ä‘Äƒng nháº­p, thÃªm auth cho `nvcr.io` vá»›i username `$oauthtoken` vÃ  NGC API
key vÃ o cÃ¹ng cáº¥u hÃ¬nh. KhÃ´ng hard-code credential trong Dockerfile/build arg.
KhÃ´ng cáº§n SSH forwarding, private package credential hoáº·c build secret.

Äá»ƒ thay base Ä‘Ã£ Ä‘Æ°á»£c kiá»ƒm tra vá»›i driver, thÃªm option sau vÃ o lá»‡nh build:

```bash
--opt build-arg:BASE_IMAGE=nvcr.io/nvidia/pytorch:TAG_DA_KIEM_TRA
```

NGC image lá»›n; cáº§n disk cho base, native snapshots vÃ  tar Ä‘áº§u ra. Dependency
Ä‘Æ°á»£c cache theo layer; native snapshotter cÃ³ thá»ƒ tá»‘n disk vÃ  thá»i gian hÆ¡n.
Äá»ƒ tÃ¡i láº­p cháº·t cháº½ hÆ¡n, pin base báº±ng digest vÃ  khÃ³a dependency sau khi build
thÃ nh cÃ´ng; requirements hiá»‡n dÃ¹ng khoáº£ng phiÃªn báº£n, khÃ´ng pháº£i lockfile.

### 4. Load vÃ  cháº¡y trÃªn mÃ¡y cÃ³ Docker/GPU

ÄÃ¢y lÃ  bÆ°á»›c bÃªn váº­n hÃ nh lÃ m trÃªn mÃ¡y cháº¡y cuá»‘i cÃ¹ng, khÃ´ng pháº£i yÃªu cáº§u
Docker daemon trÃªn Vast.ai build host. Host cáº§n NVIDIA driver tÆ°Æ¡ng thÃ­ch vÃ 
GPU container runtime Ä‘Ã£ cáº¥u hÃ¬nh. KhÃ´ng cÃ i driver NVIDIA vÃ o image.

```bash
docker load -i /path/to/clearair-wavelet.tar
docker run --rm --gpus all clearair-wavelet:latest \
  python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Náº¿u push Docker Hub, thay load báº±ng pull vÃ  dÃ¹ng tÃªn image Ä‘áº§y Ä‘á»§:

```bash
docker pull docker.io/YOUR_DOCKERHUB_USERNAME/clearair-wavelet:latest
```

#### Train cáº£ ITS vÃ  OTS, táº¡o 2 checkpoint + 6 biá»ƒu Ä‘á»“ + 4 XLSX

Runner `solution.wavelet_dehaze.run_its_ots` train tuáº§n tá»± ITS rá»“i OTS,
khÃ´ng train hai job cÃ¹ng lÃºc trÃªn má»™t GPU. Thiáº¿t láº­p theo notebook: tá»‘i Ä‘a
1000 epochs, batch 16, cosine, ITS lr=2e-4, OTS lr=1e-4, augmentation vÃ 
multi-scale. Bá»• sung validation 10% theo nhÃ³m clear image, AMP vÃ  micro-batch
4; early stopping cÃ³ thá»ƒ káº¿t thÃºc trÆ°á»›c 1000 epochs. CÃ³ thá»ƒ Ä‘á»•i qua CLI.
Runner tá»± tÃ¬m thÆ° má»¥c ghÃ©p cáº·p duy nháº¥t náº¿u archive lá»“ng má»™t cáº¥p; náº¿u cÃ³ nhiá»u
thÆ° má»¥c `hazy/clear`, pháº£i truyá»n Ä‘Æ°á»ng dáº«n chÃ­nh xÃ¡c. NÃ³ kiá»ƒm tra cáº£ bá»‘n bá»™
test trÆ°á»›c khi train Ä‘á»ƒ trÃ¡nh cháº¡y dÃ i rá»“i má»›i phÃ¡t hiá»‡n thiáº¿u dá»¯ liá»‡u.

```bash
mkdir -p /root/clearair-output
docker run --rm --gpus all --shm-size=8g \
  --mount type=bind,src=/root/clearair-data,dst=/workspace/ClearAIR/dataset,readonly \
  --mount type=bind,src=/root/clearair-output,dst=/workspace/ClearAIR/outputs \
  clearair-wavelet:latest \
  python -m solution.wavelet_dehaze.run_its_ots \
  --its-data /workspace/ClearAIR/dataset/ITS \
  --ots-data /workspace/ClearAIR/dataset/OTS \
  --eval-data-root /workspace/ClearAIR/dataset \
  --output-dir /workspace/ClearAIR/outputs/hazewavenet \
  --val-fraction 0.1 --device cuda --epochs 1000 --size 256 \
  --batch-size 16 --micro-batch-size 4 --num-workers 4 \
  --pin-memory --persistent-workers --amp
```

CÃ³ thá»ƒ cháº¡y trá»±c tiáº¿p command Python trÃªn Vast.ai hiá»‡n táº¡i náº¿u mÃ´i trÆ°á»ng
Ä‘Ã£ cÃ i Ä‘á»§ torch/CUDA vÃ  `requirements-container.txt`, khÃ´ng cáº§n cháº¡y container
lá»“ng nhau. DÃ¹ng Ä‘Æ°á»ng dáº«n host thay cho `/workspace/ClearAIR/dataset`.

Káº¿t quáº£ trong `/root/clearair-output/hazewavenet/`:

```text
haze_wavelet_its.pt
haze_wavelet_ots.pt
history_its.json
history_ots.json
its_loss.png, its_psnr.png, its_ssim.png
ots_loss.png, ots_psnr.png, ots_ssim.png
reports/hazewavenet_dataset.xlsx
reports/hazewavenet_domain.xlsx
reports/hazewavenet_fog.xlsx
reports/hardware.xlsx
reports/performance.json
```

Má»—i PNG cÃ³ Ä‘Æ°á»ng train/validation. PSNR/SSIM history lÃ  metric cá»§a trainer;
SSIM history lÃ  global SSIM, khÃ´ng giá»‘ng sliding-window SSIM á»Ÿ evaluation.
XLSX lÃ  káº¿t quáº£ inference trÃªn test, khÃ´ng pháº£i metric history train.
Checkpoint ITS dÃ¹ng cho SOTS-Indoor vÃ  I-HAZE; checkpoint OTS dÃ¹ng cho
SOTS-Outdoor vÃ  O-HAZY. ÄÃ¢y lÃ  Ä‘Ã¡nh giÃ¡ kháº£ nÄƒng tá»•ng quÃ¡t hÃ³a, **khÃ´ng pháº£i**
pipeline notebook gá»‘c train/fine-tune thÃªm hai model trÃªn I-HAZE/O-HAZY.
CÃ¡c nhÃ³m khÃ´ng cÃ³ áº£nh (vÃ­ dá»¥ má»©c sÆ°Æ¡ng nÃ o Ä‘Ã³) sáº½ Ä‘á»ƒ trá»‘ng trong Excel.
`hardware.xlsx` nay là workbook chi ti?t, g?m các sheet `PerImage`,
`DatasetSummary`, `ModelComplexity` và `Hardware`; có GMACs/GFLOPs, kích
thu?c model và stage timing cho t?ng ?nh.

Äiá»u chá»‰nh Ä‘Æ°á»ng dáº«n theo káº¿t quáº£ `find` bÆ°á»›c 2. Batch/workers/shm lÃ  cáº¥u hÃ¬nh
khá»Ÿi Ä‘áº§u, khÃ´ng pháº£i thÃ´ng sá»‘ benchmark H100 Ä‘Ã£ xÃ¡c minh. KhÃ´ng cáº§n `-p`,
environment secret, `--privileged` hoáº·c `--ipc=host`. Chá»‰ báº­t `--multi-gpu`
khi cÃ³ Ã­t nháº¥t hai GPU; code hiá»‡n dÃ¹ng DataParallel, khÃ´ng pháº£i DDP.
Output pháº£i mount writable Ä‘á»ƒ giá»¯ checkpoint sau khi container bá»‹ xÃ³a.
Trainer chá»‰ lÆ°u weights/epoch, khÃ´ng cÃ³ tÃ¹y chá»n resume optimizer/scheduler.

### Giá»›i háº¡n xÃ¡c minh

Chá»‰ static checks táº¡i workspace: source Python, COPY paths, dependency/CLI
flags, diff vÃ  shell syntax. KhÃ´ng cháº¡y `docker build`, build/push image hay
train GPU táº¡i Ä‘Ã¢y. Cáº§n xÃ¡c minh á»Ÿ Vast.ai: quyá»n pull/tag NGC, pip resolver,
quyá»n namespace BuildKit trong container thuÃª, Internet/disk, quyá»n Kaggle,
cáº¥u trÃºc archive, driver GPU vÃ  quyá»n ghi mount táº¡i mÃ¡y cháº¡y cuá»‘i cÃ¹ng.

## Cháº¡y trÃªn Google Colab

### 1. Báº­t GPU vÃ  clone repository

Trong Colab chá»n `Runtime` -> `Change runtime type` -> `T4 GPU` hoáº·c GPU máº¡nh hÆ¡n,
sau Ä‘Ã³ cháº¡y:

```python
%cd /content
!git clone https://github.com/Darkin72/VDT_GD2.git
%cd /content/VDT_GD2
```

### 2. CÃ i dependency

```python
!pip install -q torch torchvision opencv-python tqdm numpy pillow scipy \
    scikit-image psutil openpyxl
```

Kiá»ƒm tra GPU:

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 3. Kiá»ƒm tra dataset

Äáº·t dataset táº¡i `/content/VDT_GD2/dataset` theo cáº¥u trÃºc:

```text
dataset/
â”œâ”€â”€ I-HAZE/test/{clear,hazy}/
â”œâ”€â”€ O-HAZY/test/{clear,hazy}/
â””â”€â”€ Synthetic Objective Testing Set (SOTS) [RESIDE]/
    â”œâ”€â”€ indoor/test/{clear,hazy}/
    â””â”€â”€ outdoor/test/{clear,hazy}/
```

Náº¿u dataset á»Ÿ Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/dataset" /content/VDT_GD2/
```

### 4. ÄÃ¡nh giÃ¡ GridDehazeNet

Script tá»± chá»n checkpoint Ä‘Ãºng cho tá»«ng nhÃ³m:

- SOTS-Indoor vÃ  I-HAZE: `indoor_haze_best_3_6`.
- SOTS-Outdoor vÃ  O-HAZE: `outdoor_haze_best_3_6`.

```python
%cd /content/VDT_GD2
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --output-dir utils/evaluation_results/griddehazenet
```

Cháº¡y thá»­ má»™t áº£nh má»—i dataset trÆ°á»›c:

```python
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --limit 1 \
    --output-dir utils/evaluation_results/griddehazenet_smoke
```

### 5. ÄÃ¡nh giÃ¡ DCP vÃ  CAP

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

### 6. File káº¿t quáº£

Má»—i solution sinh ba file Excel:

```text
<solution>_dataset.xlsx  # SOTS-Indoor, SOTS-Outdoor, O-HAZE, I-HAZE
<solution>_domain.xlsx   # Real vÃ  Synthetic
<solution>_fog.xlsx      # Light, Medium vÃ  Heavy fog
```

Vá»›i GridDehazeNet, file tá»•ng há»£p náº±m táº¡i:

```text
utils/evaluation_results/griddehazenet/
â”œâ”€â”€ griddehazenet_dataset.xlsx
â”œâ”€â”€ griddehazenet_domain.xlsx
â”œâ”€â”€ griddehazenet_fog.xlsx
â”œâ”€â”€ summary_test.csv
â””â”€â”€ performance_test.json
```

DCP vÃ  CAP lÆ°u workbook trong thÆ° má»¥c run tÆ°Æ¡ng á»©ng:

```text
utils/evaluation_results/dcp/dcp/test/
utils/evaluation_results/cap/cap/test/
```

`performance_test.json` ghi hardware, CPU/RAM, GPU/VRAM náº¿u cÃ³, thá»i gian cháº¡y
vÃ  FPS. Trong bÃ¡o cÃ¡o nÃªn phÃ¢n biá»‡t `FPS toÃ n bá»™ run` (bao gá»“m Ä‘á»c áº£nh, resize,
tÃ­nh metric vÃ  ghi file) vá»›i `FPS suy luáº­n trung bÃ¬nh` (chá»‰ thá»i gian model).
## Download Kaggle datasets with visible progress

The sequential downloader in `scripts/download_kaggle_datasets.sh` uses
`curl --progress-bar`, validates each ZIP before extraction, and resumes an
existing archive with `curl -C -`. It downloads ITS, OTS and SOTS one at a time.

```bash
chmod +x scripts/download_kaggle_datasets.sh
bash scripts/download_kaggle_datasets.sh
```

It expects `/root/.kaggle/kaggle.json` and stores temporary archives under
`/root/clearair-data/.downloads`. Monitor another SSH session with:

```bash
watch -n 2 'du -sh /root/clearair-data/.downloads /root/clearair-data/ITS /root/clearair-data/OTS /root/clearair-data/SOTS-download 2>/dev/null; df -h /root'
```

Because ITS and SOTS are already complete in the current run, retry only OTS:

```bash
rm -f /root/clearair-data/.downloads/OTS.zip
bash scripts/download_kaggle_datasets.sh OTS
```

If the process is killed again, inspect `free -h`, `df -h /root` and
`dmesg -T | tail -n 80` for memory or disk pressure. Do not run two downloads
into the same directory.

## Authentication checklist

Before dataset download, upload a Kaggle legacy API credential to
`/root/.kaggle/kaggle.json` and run:

```bash
mkdir -p /root/.kaggle
chmod 700 /root/.kaggle
chmod 600 /root/.kaggle/kaggle.json
test -s /root/.kaggle/kaggle.json
```

The file must contain `username` and `key`. Do not commit it or copy it into
the Docker build context. Before a BuildKit push, create Docker Hub auth in
`~/.docker/config.json` using the prompt-based login script in the BuildKit
section above. BuildKit reads that file directly; Docker daemon is not needed.

## Full-image smoke training

After pulling `darkin72/dehazewavelet:latest`, run the smoke script below on a
GPU host. It creates tiny train/evaluation subsets inside the output volume,
runs one epoch for ITS and OTS, and verifies two checkpoints, two histories,
six plots, four XLSX reports and one inference PNG. It does not train on the
full dataset and its metrics are only an execution check.

```bash
chmod +x scripts/smoke_train.sh
bash scripts/smoke_train.sh
```

Optional paths/image:

```bash
IMAGE=darkin72/dehazewavelet:latest \
DATA_ROOT=/root/clearair-data \
OUTPUT_ROOT=/root/clearair-output \
bash scripts/smoke_train.sh
```

Expected results are under:

```text
/root/clearair-output/smoke-train/results/
```

## H100 batch benchmark and full four-dataset training

Do not force a batch size merely to fill exactly 80 GB. The correct setting is
the largest stable physical batch that maximizes samples/second without OOM,
NaN/Inf or a sharp throughput drop. `micro-batch-size` is the physical batch
per forward; `batch-size` is the effective batch after gradient accumulation.

Run the benchmark on the H100 host, not on GTX 1660/1080. It tests size 256
with AMP, measures allocated/reserved VRAM and samples/second, catches OOM,
and writes `batch_benchmark.json`:

```bash
chmod +x scripts/benchmark_h100_batch.sh
bash scripts/benchmark_h100_batch.sh
```

Use a smaller search first if the GPU has less VRAM:

```bash
CANDIDATES="1 2 4 8 16 32 64" \
bash scripts/benchmark_h100_batch.sh
```

Select the largest `status=ok` candidate with the highest stable
`samples_per_second`, leaving VRAM headroom. A practical initial H100 search is
`16 32 64 128 256`; the benchmark result, not the model name, decides the
final value. Keep the effective batch fixed during comparisons when studying
throughput, for example `--batch-size 128` and vary `--micro-batch-size`.

The full runner trains all four datasets using the notebook settings:

```text
I-HAZE:   500 epochs, batch 8,  lr 2e-4, step scheduler
O-HAZY:   500 epochs, batch 8,  lr 2e-4, step scheduler
SOTS-ITS: 1000 epochs, batch 16, lr 2e-4, cosine scheduler
SOTS-OTS: 1000 epochs, batch 16, lr 1e-4, cosine scheduler
```

After choosing a stable H100 micro batch, run:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,src=/root/clearair-data,dst=/workspace/ClearAIR/dataset,readonly \
  --mount type=bind,src=/root/clearair-output,dst=/workspace/ClearAIR/outputs \
  darkin72/dehazewavelet:latest \
  python -m solution.wavelet_dehaze.run_full_pipeline \
  --i-haze-data /workspace/ClearAIR/dataset/I-HAZE/train \
  --o-hazy-data /workspace/ClearAIR/dataset/O-HAZY/train \
  --its-data /workspace/ClearAIR/dataset/ITS \
  --ots-data /workspace/ClearAIR/dataset/OTS \
  --eval-data-root /workspace/ClearAIR/dataset \
  --output-dir /workspace/ClearAIR/outputs/hazewavenet \
  --device cuda \
  --size 256 \
  --micro-batch-size 32 \
  --effective-batch-size 128 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --amp \
  --augment \
  --multi-scale \
  --pin-memory \
  --persistent-workers \
  --profile-repeats 10
```

The runner creates four checkpoints and twelve curves (loss/PSNR/SSIM for
each dataset). Evaluation writes the existing three summary XLSX files plus
`hardware.xlsx`. The detailed workbook has
`PerImage`, `DatasetSummary`, `ModelComplexity` and `Hardware` sheets. It
contains preprocess time, DCP prior, DWT decomposition, low-frequency branch,
high-frequency branch, WIM reconstruction, final refinement, total inference,
PSNR and SSIM for every evaluated image, plus parameter count, MACs/FLOPs and
input sizes 256/1024.
