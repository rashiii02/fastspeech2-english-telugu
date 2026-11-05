"""Data sampling, alignment and train/val split for the English and Telugu FastSpeech 2 models.

Run from anywhere:  python scripts/pipeline.py {data,align,textgrids,split}
  data       sample 5 h train + 1 h val per language (22.05 kHz mono), build the Telugu dictionary
  align      Montreal Forced Aligner (English: pretrained, Telugu: trained from scratch)
  textgrids  copy the TextGrids to <Language>/preprocessed_data/<Language>/TextGrid/<speaker>/
  split      split train.txt / val.txt written by preprocess.py with the validation lists
Svarah is gated on Hugging Face: accept its conditions and set the HF_TOKEN environment variable.
"""
import os, io, re, sys, glob, shutil, random, argparse, subprocess, unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LANGS = ["English", "Telugu"]
SR = 22050
MIN_S, MAX_S = 1.0, 15.0
TRAIN_H, VAL_H = 5.0, 1.0

TELUGU_CFG = """beam: 100
retry_beam: 400
training:
  - monophone:
      num_iterations: 10
      max_gaussians: 1000
  - triphone:
      num_iterations: 10
      num_leaves: 2000
      max_gaussians: 10000
"""


def d_raw(lang):
    return os.path.join(ROOT, lang, "raw_data", lang)


def d_aligned(lang):
    return os.path.join(ROOT, lang, "aligned", lang)


def d_pre(lang):
    return os.path.join(ROOT, lang, "preprocessed_data", lang)


def build(ds, audio_col, name, spk_fn):
    import numpy as np
    import soundfile as sf
    import librosa
    from datasets import Audio
    ds = ds.cast_column(audio_col, Audio(decode=False))
    text_col = next(c for c in ["text", "sentence", "transcript", "transcription"] if c in ds.column_names)
    idx = list(range(len(ds)))
    random.seed(42)
    random.shuffle(idx)
    out = {"train": [], "val": []}
    hours = {"train": 0.0, "val": 0.0}
    cur = "train"
    for i in idx:
        ex = ds[i]
        y, sr = sf.read(io.BytesIO(ex[audio_col]["bytes"]))
        if y.ndim > 1:
            y = y.mean(axis=1)
        dur = len(y) / sr
        if not (MIN_S <= dur <= MAX_S):
            continue
        txt = str(ex[text_col]).strip()
        if not txt:
            continue
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
        spk = spk_fn(ex)
        d = os.path.join(d_raw(name), spk)
        os.makedirs(d, exist_ok=True)
        uid = f"{name}_{i}"
        sf.write(os.path.join(d, uid + ".wav"), y, SR)
        open(os.path.join(d, uid + ".lab"), "w", encoding="utf-8").write(txt)
        out[cur].append(f"{spk}/{uid}")
        hours[cur] += dur / 3600
        if cur == "train" and hours["train"] >= TRAIN_H:
            cur = "val"
        elif cur == "val" and hours["val"] >= VAL_H:
            break
    print(name, "train h:", round(hours["train"], 2), "val h:", round(hours["val"], 2))
    for k in out:
        open(os.path.join(ROOT, name, f"{name}_{k}_list.txt"), "w").write("\n".join(out[k]))


def make_telugu_dict():
    """Keep only the Telugu Unicode block in the transcripts and write the grapheme dictionary."""
    labs = glob.glob(os.path.join(d_raw("Telugu"), "*", "*.lab"))
    words = set()
    for p in labs:
        t = unicodedata.normalize("NFC", open(p, encoding="utf-8").read())
        t = re.sub(r"[\u200c\u200d]", "", t)
        t = re.sub(r"[^\u0C00-\u0C7F\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        open(p, "w", encoding="utf-8").write(t)
        words.update(t.split())
    with open(os.path.join(ROOT, "Telugu", "telugu.dict"), "w", encoding="utf-8") as f:
        for w in sorted(words):
            f.write(w + "\t" + " ".join(w) + "\n")
    print(len(labs), "files,", len(words), "words")


def data():
    from datasets import load_dataset
    tok = os.environ.get("HF_TOKEN")
    if tok:
        from huggingface_hub import login
        login(tok)
    svarah = load_dataset("ai4bharat/Svarah", split="test")
    telugu = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train")
    build(telugu, "audio", "Telugu", lambda ex: "male" if ex["gender"] == 1 else "female")
    build(svarah, "audio_filepath", "English", lambda ex: "spk0")
    make_telugu_dict()


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def align():
    if shutil.which("mfa") is None:
        sys.exit("mfa not found on PATH (Montreal Forced Aligner 3.4.3 is needed)")
    run(["mfa", "model", "download", "acoustic", "english_us_arpa"])
    run(["mfa", "model", "download", "dictionary", "english_us_arpa"])
    run(["mfa", "align", d_raw("English"), "english_us_arpa", "english_us_arpa", d_aligned("English"), "--clean", "-j", "4"])
    cfg = os.path.join(ROOT, "Telugu", "telugu_cfg.yaml")
    open(cfg, "w").write(TELUGU_CFG)
    run(["mfa", "train", "--config_path", cfg, "--clean", "-j", "4",
         "--output_directory", d_aligned("Telugu"), d_raw("Telugu"),
         os.path.join(ROOT, "Telugu", "telugu.dict"), os.path.join(ROOT, "Telugu", "telugu_acoustic.zip")])
    for lang in LANGS:
        print(lang, "TextGrids:", len(glob.glob(os.path.join(d_aligned(lang), "**", "*.TextGrid"), recursive=True)))


def textgrids():
    for lang in LANGS:
        spk_of = {os.path.basename(p)[:-4]: os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(d_raw(lang), "*", "*.wav"))}
        n = 0
        for tg in glob.glob(os.path.join(d_aligned(lang), "**", "*.TextGrid"), recursive=True):
            b = os.path.basename(tg)[:-9]
            if b in spk_of:
                dst = os.path.join(d_pre(lang), "TextGrid", spk_of[b])
                os.makedirs(dst, exist_ok=True)
                shutil.copy(tg, os.path.join(dst, b + ".TextGrid"))
                n += 1
        print(lang, "textgrids copied:", n)


def split():
    for lang in LANGS:
        pdir = d_pre(lang)
        val_ids = {l.strip().split("/")[-1]
                   for l in open(os.path.join(ROOT, lang, f"{lang}_val_list.txt")) if l.strip()}
        lines = []
        for f in ["train.txt", "val.txt"]:
            lines += [l for l in open(os.path.join(pdir, f), encoding="utf-8").read().splitlines() if l.strip()]
        tr = [l for l in lines if l.split("|")[0] not in val_ids]
        va = [l for l in lines if l.split("|")[0] in val_ids]
        assert tr and va, lang + ": train or val is empty"
        open(os.path.join(pdir, "train.txt"), "w", encoding="utf-8").write("\n".join(tr) + "\n")
        open(os.path.join(pdir, "val.txt"), "w", encoding="utf-8").write("\n".join(va) + "\n")
        print(lang, "train:", len(tr), "val:", len(va))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["data", "align", "textgrids", "split"])
    {"data": data, "align": align, "textgrids": textgrids, "split": split}[ap.parse_args().step]()
