# VDT-GD2: Ã„ÂÃƒÂ¡nh giÃƒÂ¡ cÃƒÂ¡c phÃ†Â°Ã†Â¡ng phÃƒÂ¡p khÃ¡Â»Â­ sÃ†Â°Ã†Â¡ng

Repository nÃƒÂ y dÃƒÂ¹ng chung pipeline Ã„â€˜ÃƒÂ¡nh giÃƒÂ¡ cho GridDehazeNet, DCP vÃƒÂ  CAP trÃƒÂªn
SOTS-Indoor, SOTS-Outdoor, O-HAZE vÃƒÂ  I-HAZE.

## Build HazeWaveNet trÃƒÂªn Vast.ai bÃ¡ÂºÂ±ng standalone BuildKit

### PhÃ¡ÂºÂ¡m vi image

Image nÃƒÂ y chÃ¡Â»Â©a mÃƒÂ´i trÃ†Â°Ã¡Â»Âng train/infer/evaluate cÃ¡Â»Â§a `solution.wavelet_dehaze`,
khÃƒÂ´ng phÃ¡ÂºÂ£i mÃƒÂ´i trÃ†Â°Ã¡Â»Âng tÃ¡Â»â€¢ng hÃ¡Â»Â£p cho mÃ¡Â»Âi model trong repo. `main.py` chÃ¡Â»â€° in lÃ¡Â»Âi
chÃƒÂ o; entrypoint train thÃ¡ÂºÂ­t lÃƒÂ  `python -m solution.wavelet_dehaze.train`.
Module dÃƒÂ¹ng relative imports nÃƒÂªn khÃƒÂ´ng chÃ¡ÂºÂ¡y trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p file `train.py`.
KhÃƒÂ´ng cÃƒÂ³ web server, port cÃ¡ÂºÂ§n expose hay environment variable application bÃ¡ÂºÂ¯t
buÃ¡Â»â„¢c. `HW_TQDM` vÃƒÂ  `HW_TQDM_BATCH` chÃ¡Â»â€° Ã„â€˜iÃ¡Â»Âu khiÃ¡Â»Æ’n progress bar.

Repo dÃƒÂ¹ng `uv` (`pyproject.toml`, `uv.lock`, Python >=3.12), nhÃ†Â°ng dependency
root thiÃ¡ÂºÂ¿u PyTorch/Pillow/numpy vÃƒÂ  cÃƒÂ¡c package evaluation. Container giÃ¡Â»Â¯
PyTorch/CUDA tÃ¡Â»Â« NGC vÃƒÂ  dÃƒÂ¹ng `pip` vÃ¡Â»â€ºi
`solution/wavelet_dehaze/requirements-container.txt` cho package cÃƒÂ²n lÃ¡ÂºÂ¡i;
khÃƒÂ´ng chÃ¡ÂºÂ¡y `uv sync` Ã„â€˜Ã¡Â»Æ’ trÃƒÂ¡nh thay thÃ¡ÂºÂ¿ bÃ¡ÂºÂ£n PyTorch GPU trong base image.
Haar DWT Ã„â€˜Ã†Â°Ã¡Â»Â£c viÃ¡ÂºÂ¿t bÃ¡ÂºÂ±ng PyTorch, khÃƒÂ´ng cÃ¡ÂºÂ§n build extension hay cÃƒÂ i OpenCV,
compiler hoÃ¡ÂºÂ·c system dependency bÃ¡Â»â€¢ sung ngoÃƒÂ i base cho phÃ¡ÂºÂ¡m vi image nÃƒÂ y.

Dockerfile cÃ…Â© Ã„â€˜Ã†Â°Ã¡Â»Â£c sÃ¡Â»Â­a ngay tÃ¡ÂºÂ¡i `solution/wavelet_dehaze/Dockerfile`: bÃ¡Â»Â script
setup khÃƒÂ´ng tÃ¡Â»â€œn tÃ¡ÂºÂ¡i, bÃ¡Â»Â UID/GID hard-code, cÃƒÂ i dependency trÃ†Â°Ã¡Â»â€ºc COPY source,
chÃ¡Â»â€° COPY HazeWaveNet vÃƒÂ  hai utility evaluation. CMD mÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh in `train --help`;
truyÃ¡Â»Ân command train khi chÃ¡ÂºÂ¡y. Image chÃ¡ÂºÂ¡y root mÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh; cÃƒÂ³ thÃ¡Â»Æ’ dÃƒÂ¹ng
`--user UID:GID` nÃ¡ÂºÂ¿u thÃ†Â° mÃ¡Â»Â¥c output trÃƒÂªn host Ã„â€˜Ã†Â°Ã¡Â»Â£c cÃ¡ÂºÂ¥p quyÃ¡Â»Ân ghi tÃ†Â°Ã†Â¡ng Ã¡Â»Â©ng.
KhÃƒÂ´ng nhÃƒÂºng dataset, checkpoint, secret hay cÃƒÂ¡c model khÃƒÂ¡c vÃƒÂ o image.

Build yÃƒÂªu cÃ¡ÂºÂ§u Linux amd64, standalone `buildkitd` Ã„â€˜ang chÃ¡ÂºÂ¡y vÃƒÂ 
`buildctl debug workers` thÃƒÂ nh cÃƒÂ´ng. Native snapshotter khÃƒÂ´ng cÃ¡ÂºÂ§n cÃƒÂº phÃƒÂ¡p
Dockerfile Ã„â€˜Ã¡ÂºÂ·c biÃ¡Â»â€¡t. KhÃƒÂ´ng cÃ¡ÂºÂ§n Docker daemon, systemd hoÃ¡ÂºÂ·c GPU khi build.
Ubuntu 22.04 cÃ¡Â»Â§a host khÃƒÂ´ng quyÃ¡ÂºÂ¿t Ã„â€˜Ã¡Â»â€¹nh phiÃƒÂªn bÃ¡ÂºÂ£n Ubuntu bÃƒÂªn trong image.
Base mÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh giÃ¡Â»Â¯ nguyÃƒÂªn `nvcr.io/nvidia/pytorch:26.04-py3` tÃ¡Â»Â« Dockerfile cÃ…Â©;
cÃ¡ÂºÂ§n xÃƒÂ¡c nhÃ¡ÂºÂ­n tag pull Ã„â€˜Ã†Â°Ã¡Â»Â£c vÃƒÂ  driver mÃƒÂ¡y chÃ¡ÂºÂ¡y tÃ†Â°Ã†Â¡ng thÃƒÂ­ch vÃ¡Â»â€ºi CUDA cÃ¡Â»Â§a base.
CÃƒÂ³ thÃ¡Â»Æ’ Ã„â€˜Ã¡Â»â€¢i bÃ¡ÂºÂ±ng build arg `BASE_IMAGE`; khÃƒÂ´ng suy ra CUDA version tÃ¡Â»Â« H100.

### 1. Git clone trÃƒÂªn Vast.ai

NhÃ¡Â»Â¯ng thay Ã„â€˜Ã¡Â»â€¢i containerization phÃ¡ÂºÂ£i cÃƒÂ³ trong revision Ã„â€˜Ã†Â°Ã¡Â»Â£c clone (workspace
nÃƒÂ y chÃ†Â°a Ã„â€˜Ã†Â°Ã¡Â»Â£c commit/push tÃ¡Â»Â± Ã„â€˜Ã¡Â»â„¢ng). MÃ¡Â»â„¢t sÃ¡Â»â€˜ dataset hiÃ¡Â»â€¡n Ã„â€˜Ã†Â°Ã¡Â»Â£c Git track;
`.dockerignore` chÃ¡Â»â€° loÃ¡ÂºÂ¡i chÃƒÂºng khÃ¡Â»Âi build context, khÃƒÂ´ng khÃ¡Â»Âi lÃ¡Â»â€¹ch sÃ¡Â»Â­ Git.
NÃ¡ÂºÂ¿u Git server hÃ¡Â»â€” trÃ¡Â»Â£ partial clone, clone source cÃ¡ÂºÂ§n thiÃ¡ÂºÂ¿t mÃƒÂ  khÃƒÂ´ng checkout
dataset bÃ¡ÂºÂ±ng cÃƒÂ¡c lÃ¡Â»â€¡nh sau:

```bash
cd /root
git clone --filter=blob:none --no-checkout https://github.com/Darkin72/VDT_GD2.git
cd VDT_GD2
git sparse-checkout init --no-cone
git sparse-checkout set /README.md /.dockerignore /scripts/ /solution/wavelet_dehaze/ /utils/
git checkout
buildctl debug workers
```

DehazeDDPM, MB-TaylorFormerV2, UDPNet Ã„â€˜ang untracked khi chuÃ¡ÂºÂ©n bÃ¡Â»â€¹ hÃ†Â°Ã¡Â»â€ºng dÃ¡ÂºÂ«n;
GridDehazeNet lÃƒÂ  gitlink. Image nÃƒÂ y khÃƒÂ´ng phÃ¡Â»Â¥ thuÃ¡Â»â„¢c nhÃ¡Â»Â¯ng thÃ†Â° mÃ¡Â»Â¥c Ã„â€˜ÃƒÂ³.

### 2. TÃ¡ÂºÂ£i dataset trÃƒÂªn Vast.ai

KhÃƒÂ´ng cÃ¡ÂºÂ§n tÃ¡ÂºÂ£i toÃƒÂ n bÃ¡Â»â„¢ dataset vÃ¡Â»Â mÃƒÂ¡y cÃƒÂ¡ nhÃƒÂ¢n. TÃ¡ÂºÂ£i vÃƒÂ o host Vast.ai, ngoÃƒÂ i repo
vÃƒÂ  ngoÃƒÂ i image. TÃ¡ÂºÂ£i cÃ¡ÂºÂ£ ITS, OTS vÃƒÂ  SOTS test theo notebook Colab:

```bash
python3 -m venv /root/kaggle-download-env
/root/kaggle-download-env/bin/python -m pip install --upgrade kaggle
mkdir -p /root/clearair-output
chmod +x scripts/download_kaggle_datasets.sh
bash scripts/download_kaggle_datasets.sh
```

NÃ¡ÂºÂ¿u thiÃ¡ÂºÂ¿u `venv`, cÃƒÂ i `python3-venv` bÃ¡ÂºÂ±ng apt trÃ†Â°Ã¡Â»â€ºc bÃ†Â°Ã¡Â»â€ºc nÃƒÂ y. CÃ¡ÂºÂ§n Internet,
quyÃ¡Â»Ân truy cÃ¡ÂºÂ­p dataset vÃƒÂ  Ã„â€˜Ã¡Â»Â§ disk cho archive lÃ¡ÂºÂ«n dÃ¡Â»Â¯ liÃ¡Â»â€¡u giÃ¡ÂºÂ£i nÃƒÂ©n. NÃ¡ÂºÂ¿u Kaggle
yÃƒÂªu cÃ¡ÂºÂ§u Ã„â€˜Ã„Æ’ng nhÃ¡ÂºÂ­p, cung cÃ¡ÂºÂ¥p credential `/root/.kaggle/kaggle.json` vÃƒÂ  Ã„â€˜Ã¡ÂºÂ·t
`chmod 600 /root/.kaggle/kaggle.json`; khÃƒÂ´ng Ã„â€˜Ã†Â°a credential vÃƒÂ o repo/image.
KhÃƒÂ´ng dÃƒÂ¹ng `datasets list --max-size 5`: giÃ¡Â»â€ºi hÃ¡ÂºÂ¡n Ã„â€˜ÃƒÂ³ tÃƒÂ­nh theo bytes.

Archive cÃƒÂ³ thÃ¡Â»Æ’ lÃ¡Â»â€œng thÃ†Â° mÃ¡Â»Â¥c. Positional argument `data` phÃ¡ÂºÂ£i lÃƒÂ  thÃ†Â° mÃ¡Â»Â¥c cha
trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p chÃ¡Â»Â©a cÃ¡ÂºÂ£ `hazy/` vÃƒÂ  `clear/`. VÃƒÂ­ dÃ¡Â»Â¥ nÃ¡ÂºÂ¿u kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ `find` lÃƒÂ 
`/root/clearair-data/ITS/ITS/hazy`, Ã„â€˜Ã†Â°Ã¡Â»Âng dÃ¡ÂºÂ«n container cÃ¡ÂºÂ§n truyÃ¡Â»Ân sÃ¡ÂºÂ½ lÃƒÂ 
`/workspace/ClearAIR/dataset/ITS/ITS`, khÃƒÂ´ng phÃ¡ÂºÂ£i `dataset/ITS`.
KhÃƒÂ´ng giÃ¡ÂºÂ£ Ã„â€˜Ã¡Â»â€¹nh tÃ¡ÂºÂ£i Kaggle tÃ¡Â»Â± tÃ¡ÂºÂ¡o train/val split. VÃ¡Â»â€ºi ITS chÃ†Â°a chia split,
dÃƒÂ¹ng `--val-fraction 0.1`: trainer chia theo clear image Ã„â€˜Ã¡Â»Æ’ trÃƒÂ¡nh leakage.
VÃ¡Â»â€ºi I-HAZE Ã„â€˜ÃƒÂ£ chia, truyÃ¡Â»Ân `I-HAZE/train` vÃƒÂ  `--val-data .../I-HAZE/val`.

```text
<data>/hazy/<Ã¡ÂºÂ£nh sÃ†Â°Ã†Â¡ng>.png
<data>/clear/<Ã¡ÂºÂ£nh rÃƒÂµ>.png
```

Trainer hÃ¡Â»â€” trÃ¡Â»Â£ tÃƒÂªn giÃ¡Â»â€˜ng nhau, hÃ¡ÂºÂ­u tÃ¡Â»â€˜ `_hazy` vÃƒÂ  biÃ¡ÂºÂ¿n thÃ¡Â»Æ’ tÃƒÂªn RESIDE.
Train tÃ¡Â»Â« Ã„â€˜Ã¡ÂºÂ§u khÃƒÂ´ng cÃ¡ÂºÂ§n checkpoint. Inference chÃ¡ÂºÂ¥t lÃ†Â°Ã¡Â»Â£ng cÃ¡ÂºÂ§n mount checkpoint
Ã„â€˜ÃƒÂ£ train vÃƒÂ  truyÃ¡Â»Ân `--checkpoint PATH`. Evaluate nhÃ¡ÂºÂ­n checkpoint qua
`--checkpoint-i-haze`, `--checkpoint-o-hazy`, `--checkpoint-sots-its`,
`--checkpoint-sots-ots`; mÃ¡Â»â„¢t file cÃƒÂ³ thÃ¡Â»Æ’ dÃƒÂ¹ng cho nhiÃ¡Â»Âu bÃ¡Â»â„¢ test.

Ã„ÂÃ¡Â»Æ’ cÃƒÂ³ Ã„â€˜Ã¡Â»Â§ 4 XLSX, cÃ¡ÂºÂ§n test trÃƒÂªn SOTS-Indoor, SOTS-Outdoor, I-HAZE vÃƒÂ  O-HAZY,
khÃƒÂ´ng chÃ¡Â»â€° train ITS/OTS. I-HAZE/O-HAZY test Ã„â€˜ÃƒÂ£ cÃƒÂ³ trong repo; sparse clone
bÃ†Â°Ã¡Â»â€ºc 1 chÃ†Â°a checkout chÃƒÂºng. LÃ¡ÂºÂ¥y riÃƒÂªng hai bÃ¡Â»â„¢ test trÃƒÂªn Vast.ai:

```bash
cd /root/VDT_GD2
git sparse-checkout add /dataset/I-HAZE/test/ /dataset/O-HAZY/test/
cp -a dataset/I-HAZE /root/clearair-data/
cp -a dataset/O-HAZY /root/clearair-data/
```

Ã„ÂÃ¡ÂºÂ·t SOTS Ã„â€˜ÃƒÂ£ giÃ¡ÂºÂ£i nÃƒÂ©n vÃƒÂ o cÃ¡ÂºÂ¥u trÃƒÂºc dÃ†Â°Ã¡Â»â€ºi Ã„â€˜ÃƒÂ¢y. NÃ¡ÂºÂ¿u archive chÃ¡Â»â€° cÃƒÂ³
`indoor/{clear,hazy}` vÃƒÂ  `outdoor/{clear,hazy}`, tÃ¡ÂºÂ¡o symlink tÃ†Â°Ã†Â¡ng Ã„â€˜Ã¡Â»â€˜i nhÃ†Â°
notebook. Thay `SOTS_SOURCE` bÃ¡ÂºÂ±ng thÃ†Â° mÃ¡Â»Â¥c chÃ¡Â»Â©a trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p `indoor/` vÃƒÂ  `outdoor/`:

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

Symlink phÃ¡ÂºÂ£i trÃ¡Â»Â bÃƒÂªn trong `/root/clearair-data` Ã„â€˜Ã¡Â»Æ’ vÃ¡ÂºÂ«n dÃƒÂ¹ng Ã„â€˜Ã†Â°Ã¡Â»Â£c khi mount.
CÃ¡ÂºÂ¥u trÃƒÂºc test cuÃ¡Â»â€˜i cÃƒÂ¹ng:

```text
/root/clearair-data/I-HAZE/test/{clear,hazy}/
/root/clearair-data/O-HAZY/test/{clear,hazy}/
/root/clearair-data/Synthetic Objective Testing Set (SOTS) [RESIDE]/indoor/test/{clear,hazy}/
/root/clearair-data/Synthetic Objective Testing Set (SOTS) [RESIDE]/outdoor/test/{clear,hazy}/
```

**Dataset Ã„â€˜ÃƒÂ£ tÃ¡ÂºÂ£i khÃƒÂ´ng tÃ¡Â»Â± Ã„â€˜i theo image khi push/pull.** MÃƒÂ¡y chÃ¡ÂºÂ¡y cuÃ¡Â»â€˜i cÃƒÂ¹ng
phÃ¡ÂºÂ£i tÃ¡ÂºÂ£i dataset hoÃ¡ÂºÂ·c nhÃ¡ÂºÂ­n bÃ¡ÂºÂ£n copy rÃ¡Â»â€œi mount nÃƒÂ³. TÃ¡ÂºÂ£i trÃ†Â°Ã¡Â»â€ºc build theo thÃ¡Â»Â© tÃ¡Â»Â±
nÃƒÂ y lÃƒÂ  Ã„â€˜Ã¡Â»Æ’ chuÃ¡ÂºÂ©n bÃ¡Â»â€¹ dÃ¡Â»Â¯ liÃ¡Â»â€¡u, khÃƒÂ´ng phÃ¡ÂºÂ£i Ã„â€˜Ã¡Â»Æ’ COPY dÃ¡Â»Â¯ liÃ¡Â»â€¡u vÃƒÂ o image.

### 3. Build image bÃ¡ÂºÂ±ng buildctl

ChÃ¡ÂºÂ¡y tÃ¡Â»Â« root repo. Dockerfile nÃ¡ÂºÂ±m trong thÃ†Â° mÃ¡Â»Â¥c con nÃƒÂªn phÃ¡ÂºÂ£i truyÃ¡Â»Ân `filename`
tÃ†Â°Ã†Â¡ng Ã„â€˜Ã¡Â»â€˜i vÃ¡Â»â€ºi `--local dockerfile=.`; `.dockerignore` nÃ¡ÂºÂ±m Ã¡Â»Å¸ root build context.

XuÃ¡ÂºÂ¥t Docker archive, khÃƒÂ´ng cÃ¡ÂºÂ§n Docker daemon:

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

#### Ã„ÂÃ„Æ’ng nhÃ¡ÂºÂ­p Docker Hub cho standalone BuildKit

TrÃ†Â°Ã¡Â»â€ºc khi push, tÃ¡ÂºÂ¡o access token Docker Hub cÃƒÂ³ quyÃ¡Â»Ân ghi repository vÃƒÂ  chÃ¡ÂºÂ¡y
script Ã„â€˜Ã„Æ’ng nhÃ¡ÂºÂ­p bÃƒÂªn dÃ†Â°Ã¡Â»â€ºi. Ã„ÂÃƒÂ¢y lÃƒÂ  auth cho BuildKit, khÃƒÂ´ng cÃ¡ÂºÂ§n Docker daemon.
Token khÃƒÂ´ng Ã„â€˜Ã†Â°Ã¡Â»Â£c ghi vÃƒÂ o source, Dockerfile hay history cÃ¡Â»Â§a shell.

HoÃ¡ÂºÂ·c push trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p lÃƒÂªn Docker Hub sau khi Ã„â€˜Ã„Æ’ng nhÃ¡ÂºÂ­p (thay username):

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

BuildKit dÃƒÂ¹ng registry credential trong `$DOCKER_CONFIG/config.json`, mÃ¡ÂºÂ·c Ã„â€˜Ã¡Â»â€¹nh
`~/.docker/config.json`. KhÃƒÂ´ng cÃ¡ÂºÂ§n chÃ¡ÂºÂ¡y `docker login`; tÃ¡ÂºÂ¡o auth bÃ¡ÂºÂ±ng prompt
Ã„â€˜Ã¡Â»Æ’ token khÃƒÂ´ng vÃƒÂ o command history:

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

Script Ã„â€˜Ã¡Â»Âc tÃ¡Â»Â« `/dev/tty` vÃƒÂ¬ stdin Ã„â€˜ang chÃ¡Â»Â©a Python heredoc. TrÃƒÂªn mÃƒÂ¡y cÃƒÂ³ Docker
CLI cÃƒÂ³ thÃ¡Â»Æ’ dÃƒÂ¹ng `docker login --username YOUR_DOCKERHUB_USERNAME`, nhÃ†Â°ng
khÃƒÂ´ng cÃ¡ÂºÂ§n lÃ¡Â»â€¡nh Ã„â€˜ÃƒÂ³ trÃƒÂªn Vast.ai. Docker CLI login cÃ…Â©ng khÃƒÂ´ng cÃ¡ÂºÂ§n daemon.

Auth lÃƒÂ  base64, khÃƒÂ´ng phÃ¡ÂºÂ£i mÃƒÂ£ hÃƒÂ³a; giÃ¡Â»Â¯ file ngoÃƒÂ i repo vÃƒÂ  khÃƒÂ´ng chia sÃ¡ÂºÂ».
NÃ¡ÂºÂ¿u cÃ¡ÂºÂ¥u hÃƒÂ¬nh cÃƒÂ³ credential helper, helper Ã„â€˜ÃƒÂ³ phÃ¡ÂºÂ£i cÃƒÂ³ trÃƒÂªn host. NÃ¡ÂºÂ¿u NGC yÃƒÂªu
cÃ¡ÂºÂ§u Ã„â€˜Ã„Æ’ng nhÃ¡ÂºÂ­p, thÃƒÂªm auth cho `nvcr.io` vÃ¡Â»â€ºi username `$oauthtoken` vÃƒÂ  NGC API
key vÃƒÂ o cÃƒÂ¹ng cÃ¡ÂºÂ¥u hÃƒÂ¬nh. KhÃƒÂ´ng hard-code credential trong Dockerfile/build arg.
KhÃƒÂ´ng cÃ¡ÂºÂ§n SSH forwarding, private package credential hoÃ¡ÂºÂ·c build secret.

Ã„ÂÃ¡Â»Æ’ thay base Ã„â€˜ÃƒÂ£ Ã„â€˜Ã†Â°Ã¡Â»Â£c kiÃ¡Â»Æ’m tra vÃ¡Â»â€ºi driver, thÃƒÂªm option sau vÃƒÂ o lÃ¡Â»â€¡nh build:

```bash
--opt build-arg:BASE_IMAGE=nvcr.io/nvidia/pytorch:TAG_DA_KIEM_TRA
```

NGC image lÃ¡Â»â€ºn; cÃ¡ÂºÂ§n disk cho base, native snapshots vÃƒÂ  tar Ã„â€˜Ã¡ÂºÂ§u ra. Dependency
Ã„â€˜Ã†Â°Ã¡Â»Â£c cache theo layer; native snapshotter cÃƒÂ³ thÃ¡Â»Æ’ tÃ¡Â»â€˜n disk vÃƒÂ  thÃ¡Â»Âi gian hÃ†Â¡n.
Ã„ÂÃ¡Â»Æ’ tÃƒÂ¡i lÃ¡ÂºÂ­p chÃ¡ÂºÂ·t chÃ¡ÂºÂ½ hÃ†Â¡n, pin base bÃ¡ÂºÂ±ng digest vÃƒÂ  khÃƒÂ³a dependency sau khi build
thÃƒÂ nh cÃƒÂ´ng; requirements hiÃ¡Â»â€¡n dÃƒÂ¹ng khoÃ¡ÂºÂ£ng phiÃƒÂªn bÃ¡ÂºÂ£n, khÃƒÂ´ng phÃ¡ÂºÂ£i lockfile.

### 4. Load vÃƒÂ  chÃ¡ÂºÂ¡y trÃƒÂªn mÃƒÂ¡y cÃƒÂ³ Docker/GPU

Ã„ÂÃƒÂ¢y lÃƒÂ  bÃ†Â°Ã¡Â»â€ºc bÃƒÂªn vÃ¡ÂºÂ­n hÃƒÂ nh lÃƒÂ m trÃƒÂªn mÃƒÂ¡y chÃ¡ÂºÂ¡y cuÃ¡Â»â€˜i cÃƒÂ¹ng, khÃƒÂ´ng phÃ¡ÂºÂ£i yÃƒÂªu cÃ¡ÂºÂ§u
Docker daemon trÃƒÂªn Vast.ai build host. Host cÃ¡ÂºÂ§n NVIDIA driver tÃ†Â°Ã†Â¡ng thÃƒÂ­ch vÃƒÂ 
GPU container runtime Ã„â€˜ÃƒÂ£ cÃ¡ÂºÂ¥u hÃƒÂ¬nh. KhÃƒÂ´ng cÃƒÂ i driver NVIDIA vÃƒÂ o image.

```bash
docker load -i /path/to/clearair-wavelet.tar
docker run --rm --gpus all clearair-wavelet:latest \
  python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

NÃ¡ÂºÂ¿u push Docker Hub, thay load bÃ¡ÂºÂ±ng pull vÃƒÂ  dÃƒÂ¹ng tÃƒÂªn image Ã„â€˜Ã¡ÂºÂ§y Ã„â€˜Ã¡Â»Â§:

```bash
docker pull docker.io/YOUR_DOCKERHUB_USERNAME/clearair-wavelet:latest
```

#### Train cÃ¡ÂºÂ£ ITS vÃƒÂ  OTS, tÃ¡ÂºÂ¡o 2 checkpoint + 6 biÃ¡Â»Æ’u Ã„â€˜Ã¡Â»â€œ + 4 XLSX

Runner `solution.wavelet_dehaze.run_its_ots` train tuÃ¡ÂºÂ§n tÃ¡Â»Â± ITS rÃ¡Â»â€œi OTS,
khÃƒÂ´ng train hai job cÃƒÂ¹ng lÃƒÂºc trÃƒÂªn mÃ¡Â»â„¢t GPU. ThiÃ¡ÂºÂ¿t lÃ¡ÂºÂ­p theo notebook: tÃ¡Â»â€˜i Ã„â€˜a
1000 epochs, batch 16, cosine, ITS lr=2e-4, OTS lr=1e-4, augmentation vÃƒÂ 
multi-scale. BÃ¡Â»â€¢ sung validation 10% theo nhÃƒÂ³m clear image, AMP vÃƒÂ  micro-batch
4; early stopping cÃƒÂ³ thÃ¡Â»Æ’ kÃ¡ÂºÂ¿t thÃƒÂºc trÃ†Â°Ã¡Â»â€ºc 1000 epochs. CÃƒÂ³ thÃ¡Â»Æ’ Ã„â€˜Ã¡Â»â€¢i qua CLI.
Runner tÃ¡Â»Â± tÃƒÂ¬m thÃ†Â° mÃ¡Â»Â¥c ghÃƒÂ©p cÃ¡ÂºÂ·p duy nhÃ¡ÂºÂ¥t nÃ¡ÂºÂ¿u archive lÃ¡Â»â€œng mÃ¡Â»â„¢t cÃ¡ÂºÂ¥p; nÃ¡ÂºÂ¿u cÃƒÂ³ nhiÃ¡Â»Âu
thÃ†Â° mÃ¡Â»Â¥c `hazy/clear`, phÃ¡ÂºÂ£i truyÃ¡Â»Ân Ã„â€˜Ã†Â°Ã¡Â»Âng dÃ¡ÂºÂ«n chÃƒÂ­nh xÃƒÂ¡c. NÃƒÂ³ kiÃ¡Â»Æ’m tra cÃ¡ÂºÂ£ bÃ¡Â»â€˜n bÃ¡Â»â„¢
test trÃ†Â°Ã¡Â»â€ºc khi train Ã„â€˜Ã¡Â»Æ’ trÃƒÂ¡nh chÃ¡ÂºÂ¡y dÃƒÂ i rÃ¡Â»â€œi mÃ¡Â»â€ºi phÃƒÂ¡t hiÃ¡Â»â€¡n thiÃ¡ÂºÂ¿u dÃ¡Â»Â¯ liÃ¡Â»â€¡u.

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

CÃƒÂ³ thÃ¡Â»Æ’ chÃ¡ÂºÂ¡y trÃ¡Â»Â±c tiÃ¡ÂºÂ¿p command Python trÃƒÂªn Vast.ai hiÃ¡Â»â€¡n tÃ¡ÂºÂ¡i nÃ¡ÂºÂ¿u mÃƒÂ´i trÃ†Â°Ã¡Â»Âng
Ã„â€˜ÃƒÂ£ cÃƒÂ i Ã„â€˜Ã¡Â»Â§ torch/CUDA vÃƒÂ  `requirements-container.txt`, khÃƒÂ´ng cÃ¡ÂºÂ§n chÃ¡ÂºÂ¡y container
lÃ¡Â»â€œng nhau. DÃƒÂ¹ng Ã„â€˜Ã†Â°Ã¡Â»Âng dÃ¡ÂºÂ«n host thay cho `/workspace/ClearAIR/dataset`.

KÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ trong `/root/clearair-output/hazewavenet/`:

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

MÃ¡Â»â€”i PNG cÃƒÂ³ Ã„â€˜Ã†Â°Ã¡Â»Âng train/validation. PSNR/SSIM history lÃƒÂ  metric cÃ¡Â»Â§a trainer;
SSIM history lÃƒÂ  global SSIM, khÃƒÂ´ng giÃ¡Â»â€˜ng sliding-window SSIM Ã¡Â»Å¸ evaluation.
XLSX lÃƒÂ  kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ inference trÃƒÂªn test, khÃƒÂ´ng phÃ¡ÂºÂ£i metric history train.
Checkpoint ITS dÃƒÂ¹ng cho SOTS-Indoor vÃƒÂ  I-HAZE; checkpoint OTS dÃƒÂ¹ng cho
SOTS-Outdoor vÃƒÂ  O-HAZY. Ã„ÂÃƒÂ¢y lÃƒÂ  Ã„â€˜ÃƒÂ¡nh giÃƒÂ¡ khÃ¡ÂºÂ£ nÃ„Æ’ng tÃ¡Â»â€¢ng quÃƒÂ¡t hÃƒÂ³a, **khÃƒÂ´ng phÃ¡ÂºÂ£i**
pipeline notebook gÃ¡Â»â€˜c train/fine-tune thÃƒÂªm hai model trÃƒÂªn I-HAZE/O-HAZY.
CÃƒÂ¡c nhÃƒÂ³m khÃƒÂ´ng cÃƒÂ³ Ã¡ÂºÂ£nh (vÃƒÂ­ dÃ¡Â»Â¥ mÃ¡Â»Â©c sÃ†Â°Ã†Â¡ng nÃƒÂ o Ã„â€˜ÃƒÂ³) sÃ¡ÂºÂ½ Ã„â€˜Ã¡Â»Æ’ trÃ¡Â»â€˜ng trong Excel.
`hardware.xlsx` nay lÃ  workbook chi ti?t, g?m cÃ¡c sheet `PerImage`,
`DatasetSummary`, `ModelComplexity` vÃ  `Hardware`; cÃ³ GMACs/GFLOPs, kÃ­ch
thu?c model vÃ  stage timing cho t?ng ?nh.

Ã„ÂiÃ¡Â»Âu chÃ¡Â»â€°nh Ã„â€˜Ã†Â°Ã¡Â»Âng dÃ¡ÂºÂ«n theo kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£ `find` bÃ†Â°Ã¡Â»â€ºc 2. Batch/workers/shm lÃƒÂ  cÃ¡ÂºÂ¥u hÃƒÂ¬nh
khÃ¡Â»Å¸i Ã„â€˜Ã¡ÂºÂ§u, khÃƒÂ´ng phÃ¡ÂºÂ£i thÃƒÂ´ng sÃ¡Â»â€˜ benchmark H100 Ã„â€˜ÃƒÂ£ xÃƒÂ¡c minh. KhÃƒÂ´ng cÃ¡ÂºÂ§n `-p`,
environment secret, `--privileged` hoÃ¡ÂºÂ·c `--ipc=host`. ChÃ¡Â»â€° bÃ¡ÂºÂ­t `--multi-gpu`
khi cÃƒÂ³ ÃƒÂ­t nhÃ¡ÂºÂ¥t hai GPU; code hiÃ¡Â»â€¡n dÃƒÂ¹ng DataParallel, khÃƒÂ´ng phÃ¡ÂºÂ£i DDP.
Output phÃ¡ÂºÂ£i mount writable Ã„â€˜Ã¡Â»Æ’ giÃ¡Â»Â¯ checkpoint sau khi container bÃ¡Â»â€¹ xÃƒÂ³a.
Trainer chÃ¡Â»â€° lÃ†Â°u weights/epoch, khÃƒÂ´ng cÃƒÂ³ tÃƒÂ¹y chÃ¡Â»Ân resume optimizer/scheduler.

### GiÃ¡Â»â€ºi hÃ¡ÂºÂ¡n xÃƒÂ¡c minh

ChÃ¡Â»â€° static checks tÃ¡ÂºÂ¡i workspace: source Python, COPY paths, dependency/CLI
flags, diff vÃƒÂ  shell syntax. KhÃƒÂ´ng chÃ¡ÂºÂ¡y `docker build`, build/push image hay
train GPU tÃ¡ÂºÂ¡i Ã„â€˜ÃƒÂ¢y. CÃ¡ÂºÂ§n xÃƒÂ¡c minh Ã¡Â»Å¸ Vast.ai: quyÃ¡Â»Ân pull/tag NGC, pip resolver,
quyÃ¡Â»Ân namespace BuildKit trong container thuÃƒÂª, Internet/disk, quyÃ¡Â»Ân Kaggle,
cÃ¡ÂºÂ¥u trÃƒÂºc archive, driver GPU vÃƒÂ  quyÃ¡Â»Ân ghi mount tÃ¡ÂºÂ¡i mÃƒÂ¡y chÃ¡ÂºÂ¡y cuÃ¡Â»â€˜i cÃƒÂ¹ng.

## ChÃ¡ÂºÂ¡y trÃƒÂªn Google Colab

### 1. BÃ¡ÂºÂ­t GPU vÃƒÂ  clone repository

Trong Colab chÃ¡Â»Ân `Runtime` -> `Change runtime type` -> `T4 GPU` hoÃ¡ÂºÂ·c GPU mÃ¡ÂºÂ¡nh hÃ†Â¡n,
sau Ã„â€˜ÃƒÂ³ chÃ¡ÂºÂ¡y:

```python
%cd /content
!git clone https://github.com/Darkin72/VDT_GD2.git
%cd /content/VDT_GD2
```

### 2. CÃƒÂ i dependency

```python
!pip install -q torch torchvision opencv-python tqdm numpy pillow scipy \
    scikit-image psutil openpyxl
```

KiÃ¡Â»Æ’m tra GPU:

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

### 3. KiÃ¡Â»Æ’m tra dataset

Ã„ÂÃ¡ÂºÂ·t dataset tÃ¡ÂºÂ¡i `/content/VDT_GD2/dataset` theo cÃ¡ÂºÂ¥u trÃƒÂºc:

```text
dataset/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ I-HAZE/test/{clear,hazy}/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ O-HAZY/test/{clear,hazy}/
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ Synthetic Objective Testing Set (SOTS) [RESIDE]/
    Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ indoor/test/{clear,hazy}/
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ outdoor/test/{clear,hazy}/
```

NÃ¡ÂºÂ¿u dataset Ã¡Â»Å¸ Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r "/content/drive/MyDrive/dataset" /content/VDT_GD2/
```

### 4. Ã„ÂÃƒÂ¡nh giÃƒÂ¡ GridDehazeNet

Script tÃ¡Â»Â± chÃ¡Â»Ân checkpoint Ã„â€˜ÃƒÂºng cho tÃ¡Â»Â«ng nhÃƒÂ³m:

- SOTS-Indoor vÃƒÂ  I-HAZE: `indoor_haze_best_3_6`.
- SOTS-Outdoor vÃƒÂ  O-HAZE: `outdoor_haze_best_3_6`.

```python
%cd /content/VDT_GD2
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --output-dir utils/evaluation_results/griddehazenet
```

ChÃ¡ÂºÂ¡y thÃ¡Â»Â­ mÃ¡Â»â„¢t Ã¡ÂºÂ£nh mÃ¡Â»â€”i dataset trÃ†Â°Ã¡Â»â€ºc:

```python
!python utils/evaluate_griddehazenet.py \
    --device cuda \
    --limit 1 \
    --output-dir utils/evaluation_results/griddehazenet_smoke
```

### 5. Ã„ÂÃƒÂ¡nh giÃƒÂ¡ DCP vÃƒÂ  CAP

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

### 6. File kÃ¡ÂºÂ¿t quÃ¡ÂºÂ£

MÃ¡Â»â€”i solution sinh ba file Excel:

```text
<solution>_dataset.xlsx  # SOTS-Indoor, SOTS-Outdoor, O-HAZE, I-HAZE
<solution>_domain.xlsx   # Real vÃƒÂ  Synthetic
<solution>_fog.xlsx      # Light, Medium vÃƒÂ  Heavy fog
```

VÃ¡Â»â€ºi GridDehazeNet, file tÃ¡Â»â€¢ng hÃ¡Â»Â£p nÃ¡ÂºÂ±m tÃ¡ÂºÂ¡i:

```text
utils/evaluation_results/griddehazenet/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ griddehazenet_dataset.xlsx
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ griddehazenet_domain.xlsx
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ griddehazenet_fog.xlsx
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ summary_test.csv
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ performance_test.json
```

DCP vÃƒÂ  CAP lÃ†Â°u workbook trong thÃ†Â° mÃ¡Â»Â¥c run tÃ†Â°Ã†Â¡ng Ã¡Â»Â©ng:

```text
utils/evaluation_results/dcp/dcp/test/
utils/evaluation_results/cap/cap/test/
```

`performance_test.json` ghi hardware, CPU/RAM, GPU/VRAM nÃ¡ÂºÂ¿u cÃƒÂ³, thÃ¡Â»Âi gian chÃ¡ÂºÂ¡y
vÃƒÂ  FPS. Trong bÃƒÂ¡o cÃƒÂ¡o nÃƒÂªn phÃƒÂ¢n biÃ¡Â»â€¡t `FPS toÃƒÂ n bÃ¡Â»â„¢ run` (bao gÃ¡Â»â€œm Ã„â€˜Ã¡Â»Âc Ã¡ÂºÂ£nh, resize,
tÃƒÂ­nh metric vÃƒÂ  ghi file) vÃ¡Â»â€ºi `FPS suy luÃ¡ÂºÂ­n trung bÃƒÂ¬nh` (chÃ¡Â»â€° thÃ¡Â»Âi gian model).
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

Ngoài workbook, evaluation in timing trung bình riêng cho từng dataset vào `timing_i_haze.json`, `timing_o_hazy.json`, `timing_sots_its.json` và `timing_sots_ots.json`; terminal cũng in toàn bộ stage timing theo từng dataset.
