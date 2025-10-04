# Fine-Tuning F5‑TTS for Arabic (Arabic Speech Corpus)

This README documents **everything needed to reproduce our Arabic fine-tuning pipeline** end-to-end on Windows: converting **Buckwalter** transcripts to **Arabic with diacritics**, preparing the dataset, wiring vocab/paths, and launching training.

> ✅ Target setup  
> - Model family: **F5‑TTS**  
> - Checkpoint: e.g. `model_380000.pt` (Arabic)  
> - Text: **Arabic (with diacritics)**, not pinyin  
> - Tokenizer: **char** (reuse **IbrahimSalah/F5‑TTS‑Arabic** `vocab.txt`)  
> - OS/GPU: Windows + single NVIDIA GPU

---

## Quickstart (copy–paste)

```cmd
:: 0) One-time Accelerate config
accelerate config

:: 1) Make sure your prepared files exist here:
:: C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\
::   raw.arrow  duration.json  vocab.txt   (Ibrahim's vocab)

:: 2) Train (sample-based, smoke test)
set PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src && ^
accelerate launch --mixed_precision=fp16 ^
  "C:\Users\User\Desktop\F5-TTS\src\f5_tts\train\finetune_cli.py" ^
  --exp_name F5TTS_v1_Base ^
  --learning_rate 1e-4 ^
  --batch_size_type sample --batch_size_per_gpu 2 ^
  --grad_accumulation_steps 4 --max_samples 2 ^
  --epochs 1 --num_warmup_updates 0 ^
  --save_per_updates 50 --keep_last_n_checkpoints -1 --last_per_updates 50 ^
  --dataset_name my_dataset_ar --finetune ^
  --pretrain "C:\Users\User\Desktop\F5-TTS\ckpts\model_380000.pt" ^
  --tokenizer char ^
  --tokenizer_path "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\vocab.txt" ^
  --log_samples
```

Once you see actual training steps (loss updates), switch to **frame** batching and scale up.

---

## Contents

- [Prerequisites](#prerequisites)  
- [Folder Layout](#folder-layout)  
- [Step 1 — Parse raw CSV to `metadata.csv`](#step-1--parse-raw-csv-to-metadatacsv)  
- [Step 2 — Buckwalter → Arabic (with diacritics)](#step-2--buckwalter--arabic-with-diacritics)  
- [Step 3 — Assemble `/your_dataset` (copy WAVs, filter 3–12s)](#step-3--assemble-your_dataset-copy-wavs-filter-312s)  
- [Step 4 — Prepare Arrow dataset (disable pinyin; use Arabic vocab)](#step-4--prepare-arrow-dataset-disable-pinyin-use-arabic-vocab)  
- [Step 5 — Dataset path convention (`_char`)](#step-5--dataset-path-convention-_char)  
- [Step 6 — Use local code, not `site-packages`](#step-6--use-local-code-not-site-packages)  
- [Step 7 — (Optional) Gradio Finetune UI](#step-7--optional-gradio-finetune-ui)  
- [Training commands](#training-commands)  
- [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)  
- [Appendix: Helper script (Buckwalter → Arabic)](#appendix-helper-script-buckwalter--arabic)

---

## Prerequisites

- Python env with: `torch`, `torchaudio`, `datasets`, `pyarrow`, `tqdm`, `accelerate`, and **FFmpeg/ffprobe** in PATH.
- F5‑TTS repo cloned to: `C:\Users\User\Desktop\F5-TTS`
- Arabic pretrained checkpoint: `C:\Users\User\Desktop\F5-TTS\ckpts\model_380000.pt`
- **IbrahimSalah/F5‑TTS‑Arabic `vocab.txt`** available

> Run once: `accelerate config`  
> - This machine → No distributed training → GPU id `0` → mixed precision `fp16` → backend `gloo` (Windows) → Save

---

## Folder Layout

```
C:\Users\User\Desktop\F5-TTS
├─ data
│  └─ my_dataset_ar_char
│     ├─ raw.arrow
│     ├─ duration.json
│     └─ vocab.txt             # Ibrahim Salah's vocab
├─ ckpts
│  └─ model_380000.pt          # pretrained Arabic checkpoint
└─ src
   └─ f5_tts
      ├─ train
      │  ├─ finetune_cli.py
      │  ├─ finetune_gradio.py         # paths adjusted to absolute Windows dirs
      │  └─ datasets
      │     └─ prepare_csv_wavs.py     # pinyin disabled; PRETRAINED_VOCAB_PATH set
      └─ model
         └─ dataset.py                 # (optional) custom data path logic
```

---

## Step 1 — Parse raw CSV to `metadata.csv`

**Original rows (Excel, one column)**:
```
"ARA NORM  0002.wav" "waraj~aHa Alt~aqoriyru ..."
```

**Goal** → pipe-delimited CSV:
```
audio_file|text
wavs/ARA NORM  0002.wav|waraj~aHa Alt~aqoriyru ...
```

Use a small parser to split `"filename" "text"` and write `audio_file|text`.  
Output: `output_keepnames.csv` (or `output_renumber.csv` if you want `audio_0001.wav` style later).

---

## Step 2 — Buckwalter → Arabic (with diacritics)

Use the helper in the appendix (or any Buckwalter mapper):

```powershell
python convert_buckwalter_to_arabic.py `
  --in  C:\path\to\output_keepnames.csv `
  --out C:\path\to\metadata_arabic.csv
```

Resulting `metadata_arabic.csv` keeps header `audio_file|text` and converts the text column to Arabic Unicode with diacritics.

---

## Step 3 — Assemble `/your_dataset` (copy WAVs, filter 3–12s)

Create a dataset folder containing your WAVs and `metadata.csv`:

```
C:\Users\User\Desktop\arabic-speech-corpus
│-- metadata.csv           # rename from metadata_arabic.csv
└─ wavs\                   # copied audio, 3–12 s per clip
```

We used a small script to copy WAVs and filter durations (requires `ffprobe`).  
(If you prefer, you can skip filtering and rely on the trainer’s duration guard.)

---

## Step 4 — Prepare Arrow dataset (disable pinyin; use Arabic vocab)

**File:** `src/f5_tts/train/datasets/prepare_csv_wavs.py`

1) **Disable pinyin conversion**:
```python
def batch_convert_texts(texts, polyphone, batch_size=BATCH_SIZE):
    return texts  # keep Arabic as-is
```

2) **Use Ibrahim’s vocab** (fine-tune mode copies a pretrained vocab):
```python
from pathlib import Path
PRETRAINED_VOCAB_PATH = Path(r"C:\path\to\IbrahimSalah-F5-TTS-Arabic\vocab.txt")
```

> ⚠️ Do **not** set this to the **output** file path; Windows will throw `WinError 32` (file in use).

**Run**:
```powershell
python .\src\f5_tts\train\datasets\prepare_csv_wavs.py `
  "C:\Users\User\Desktop\arabic-speech-corpus" `
  "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar"
```

**Output**:
```
C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar
│-- raw.arrow
│-- duration.json
└-- vocab.txt    # Ibrahim's vocab (copied)
```

If the vocab copy fails, copy manually:
```powershell
Copy-Item "C:\path\to\Ibrahim\vocab.txt" `
          "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar\vocab.txt" -Force
```

---

## Step 5 — Dataset path convention `_char`

The loader composes:
```
{path_data}\{dataset_name}_{tokenizer}
```

So for `--dataset_name my_dataset_ar` and `--tokenizer char`, place/duplicate your prepared folder as:

```
C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\
  raw.arrow  duration.json  vocab.txt
```

---

## Step 6 — Use local code, not `site-packages`

Ensure Python imports your edited repo code:

- **CMD**:
  ```cmd
  set PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src
  ```
- **PowerShell**:
  ```powershell
  $env:PYTHONPATH = "C:\Users\User\Desktop\F5-TTS\src"
  ```

(Or `pip install -e .`)

---

## Step 7 — (Optional) Gradio Finetune UI

```powershell
python .\src\f5_tts\train\finetune_gradio.py
```

Set absolute paths inside `finetune_gradio.py`:

```python
path_data = r"C:\Users\User\Desktop\F5-TTS\data"
path_project_ckpts = r"C:\Users\User\Desktop\F5-TTS\ckpts"
file_train = r"C:\Users\User\Desktop\F5-TTS\src\f5_tts\train\finetune_cli.py"
```

- **Tokenizer Type:** `char`
- **Tokenizer File:** `C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\vocab.txt`
- **Checkpoint:** `C:\Users\User\Desktop\F5-TTS\ckpts\model_380000.pt`
- **EMA:** off
- **Batch per GPU:** start small; increase gradually

For “Vocab Check” tab, copy `metadata.csv` into the prepared folder if needed.

---

## Training commands

### A) Sample-based (robust smoke test)

```cmd
set PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src && ^
accelerate launch --mixed_precision=fp16 ^
  "C:\Users\User\Desktop\F5-TTS\src\f5_tts\train\finetune_cli.py" ^
  --exp_name F5TTS_v1_Base ^
  --learning_rate 1e-4 ^
  --batch_size_type sample --batch_size_per_gpu 2 ^
  --grad_accumulation_steps 4 --max_samples 2 ^
  --epochs 1 --num_warmup_updates 0 ^
  --save_per_updates 50 --keep_last_n_checkpoints -1 --last_per_updates 50 ^
  --dataset_name my_dataset_ar --finetune ^
  --pretrain "C:\Users\User\Desktop\F5-TTS\ckpts\model_380000.pt" ^
  --tokenizer char ^
  --tokenizer_path "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\vocab.txt" ^
  --log_samples
```

### B) Frame-based (tune after sanity test)

24kHz, hop=256 → **frames ≈ seconds × 93.75**  
For up to ~12s, set frames ≥ **1300**; for longer clips, increase accordingly.

```cmd
set PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src && ^
accelerate launch --mixed_precision=fp16 ^
  "C:\Users\User\Desktop\F5-TTS\src\f5_tts\train\finetune_cli.py" ^
  --exp_name F5TTS_v1_Base ^
  --learning_rate 1e-4 ^
  --batch_size_type frame --batch_size_per_gpu 1600 ^
  --grad_accumulation_steps 2 --max_samples 0 ^
  --epochs 1 --num_warmup_updates 100 ^
  --save_per_updates 200 --keep_last_n_checkpoints -1 --last_per_updates 100 ^
  --dataset_name my_dataset_ar --finetune ^
  --pretrain "C:\Users\User\Desktop\F5-TTS\ckpts\model_380000.pt" ^
  --tokenizer char ^
  --tokenizer_path "C:\Users\User\Desktop\F5-TTS\data\my_dataset_ar_char\vocab.txt" ^
  --log_samples
```

> If you hit OOM: reduce frames to 1200 and increase `--grad_accumulation_steps`.

---

## Troubleshooting & Common Pitfalls

- **Immediate exit (“Saved last checkpoint at update 380000”)**  
  → **Zero batches** produced.  
  Fix: use **sample-based** first; for frames, raise `--batch_size_per_gpu` above your longest clip; set a small non‑zero `--max_samples` for a smoke test.

- **Package path vs repo path**  
  Seeing `...site-packages\f5_tts...` in traces means you’re not using local code.  
  Fix: set `PYTHONPATH=C:\Users\User\Desktop\F5-TTS\src` (same CMD window).

- **Vocab copy `WinError 32`**  
  You pointed `PRETRAINED_VOCAB_PATH` to the **destination**.  
  Fix: point to the **source** (Ibrahim’s), or copy manually after the run.

- **Tokenizer mismatch**  
  Use `--tokenizer char` (Arabic). Avoid pinyin conversion entirely.

- **Dataset path not found**  
  With `--dataset_name my_dataset_ar` and `--tokenizer char`, the loader expects:  
  `...\data\my_dataset_ar_char\raw.arrow` (plus `duration.json`, `vocab.txt`).

- **Accelerate warning**  
  Run `accelerate config` once; choose single‑GPU `fp16` settings.

---

## Appendix: Helper script (Buckwalter → Arabic)

Save as `convert_buckwalter_to_arabic.py`:

```python
#!/usr/bin/env python3
import argparse

BW2AR = {
    "'":"ء",">":"أ","<":"إ","&":"ؤ","}":"ئ","|":"آ",
    "A":"ا","b":"ب","t":"ت","v":"ث","j":"ج","H":"ح","x":"خ",
    "d":"د","*":"ذ","r":"ر","z":"ز","s":"س","$":"ش",
    "S":"ص","D":"ض","T":"ط","Z":"ظ","E":"ع","g":"غ",
    "f":"ف","q":"ق","k":"ك","l":"ل","m":"م","n":"ن",
    "h":"ه","w":"و","y":"ي","Y":"ى","p":"ة","_":"ـ",
    "a":"َ","u":"ُ","i":"ِ","F":"ً","N":"ٌ","K":"ٍ","o":"ْ","~":"ّ","^":"ٰ",
}

def bw2ar(text: str) -> str:
    return ''.join(BW2AR.get(ch, ch) for ch in text)

def convert(inp, outp):
    with open(inp, 'r', encoding='utf-8-sig') as fin, open(outp, 'w', encoding='utf-8', newline='') as fout:
        first = True
        for line in fin:
            if first:
                first = False
                if line.strip() != "audio_file|text":
                    fout.write("audio_file|text\n")
                    if "|" not in line:
                        continue
                else:
                    fout.write("audio_file|text\n")
                    continue
            s = line.rstrip('\n')
            if not s or "|" not in s:
                continue
            left, right = s.split("|", 1)  # split only at first pipe
            fout.write(f"{left}|{bw2ar(right)}\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='outp', required=True)
    args = ap.parse_args()
    convert(args.inp, args.outp)
```

**Run:**
```powershell
python convert_buckwalter_to_arabic.py --in C:\path\to\output_keepnames.csv --out C:\path\to\metadata_arabic.csv
```
