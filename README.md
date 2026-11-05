# FastSpeech 2 TTS: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models, one per language, built on the PyTorch implementation
https://github.com/ming024/FastSpeech2 with the HiFi-GAN universal vocoder.
`English/` and `Telugu/` each hold that implementation plus the changes listed below.

## Repository layout

    English/          code, config.yaml, checkpoints/, inference.py
    Telugu/           code, config.yaml, checkpoints/, inference.py
    Results/          audio and plots for the 5 test sentences per language
    Report/           Technical_Report.pdf
    scripts/          pipeline.py: data sampling, alignment, train/val split
    requirements.txt

Each language folder also has preprocess.yaml, model.yaml, train.yaml (config.yaml is the three merged),
preprocess.py and train.py.

## Setup

    pip install -r requirements.txt

Tested on Python 3.13 (Kaggle, 2 x T4). setuptools<81 is pinned because pyworld imports pkg_resources.

## Inference with the included checkpoints

    cd English
    python inference.py --ckpt checkpoints/English_20000_fp16.pth.tar
    cd ../Telugu
    python inference.py --ckpt checkpoints/Telugu_20000_fp16.pth.tar

Audio and the duration / pitch / energy plots are written to ../Results/<Language>/ by default
(change with --out). --speaker selects the speaker; Telugu is multi-speaker (one male, one female voice).
The first run unzips the HiFi-GAN vocoder in hifigan/. Checkpoints are weights only, stored in fp16 (68 MB each).

## Data

- English: Svarah (ai4bharat/Svarah, test split, 6656 utterances). Svarah is an ASR benchmark, not a TTS
  corpus, and has no speaker id column, so all English audio is treated as one speaker and the English model
  is single-speaker.
- Telugu: IndicTTS Telugu (SPRINGLab/IndicTTS_Telugu). One speaker has only about 4.3-4.5 h, so both
  speakers (male and female) are used and the Telugu model is multi-speaker.
- Sampling: utterances are shuffled with seed 42, kept if 1-15 s long, resampled to 22.05 kHz mono, and
  added to train until 5.0 h, then to validation until 1.0 h. Train and validation never share a file.
- After alignment and preprocessing: English train 3185 / val 637 utterances, Telugu train 2261 / val 452.
  The preprocessor reports 5.22 h (English) and 5.13 h (Telugu) in total, because unaligned utterances are
  dropped and leading/trailing silence is trimmed.

## Alignment (Montreal Forced Aligner 3.4.3)

- English: pretrained english_us_arpa acoustic model and dictionary. 3889 of 3924 utterances aligned.
- Telugu: no pretrained model exists, so an acoustic model is trained from scratch on this data with a
  grapheme dictionary (every Telugu character is one symbol). Training is shortened to 10 monophone and
  10 triphone iterations to fit the compute budget. 2713 of 2718 utterances aligned.

## Changes to the reference implementation

- Telugu text: NFC normalisation, zero-width joiners removed, only the Telugu Unicode block
  (U+0C00-U+0C7F) kept. These characters are added to text/symbols.py and the Telugu config uses
  basic_cleaners. The Telugu model is multi-speaker.
- preprocessor/preprocessor.py: reads TextGrids with include_empty_intervals=True, uses tier.intervals
  (newer tgt), and maps empty intervals to "sp".
- audio/stft.py: keyword arguments for the newer librosa.
- train.py: losses are cast to float32 (dtype mismatch with the newer PyTorch).

## Training

20000 steps per language, batch size 16, Adam (0.9, 0.98), 4000 warm-up steps, one T4 GPU per language.
Loss values are logged every 100 steps.

## Reproducing from scratch

Svarah is gated on Hugging Face: accept its conditions and set HF_TOKEN. Montreal Forced Aligner 3.4.3
must be on PATH.

    python scripts/pipeline.py data
    python scripts/pipeline.py align
    python scripts/pipeline.py textgrids
    cd English && python preprocess.py preprocess.yaml && cd ..
    cd Telugu && python preprocess.py preprocess.yaml && cd ..
    python scripts/pipeline.py split
    cd English && python train.py -p preprocess.yaml -m model.yaml -t train.yaml
    cd Telugu && python train.py -p preprocess.yaml -m model.yaml -t train.yaml

Details, loss curves and the per-sentence analysis are in Report/Technical_Report.pdf.

<!-- README END -->
