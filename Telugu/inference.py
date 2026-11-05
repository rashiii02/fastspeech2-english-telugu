"""FastSpeech 2 inference for Telugu (run from this folder: python inference.py)."""
import os, re, sys, json, yaml, glob, zipfile, unicodedata, argparse
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.io import wavfile

LANG = "Telugu"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
from text import text_to_sequence
from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer

device = "cuda" if torch.cuda.is_available() else "cpu"

EN = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}
TE = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}
SENTENCES = EN if LANG == "English" else TE
DEFAULT_SPEAKER = "spk0" if LANG == "English" else "female"

# ---------------- text frontend ----------------
EN_NORM = [("10:45 AM", "ten forty five AMX"),
           ("12 November 2026", "twelve November twenty twenty six")]
EN_OVR = {"amx": ["EY1", "EH1", "M"]}
TE_NORM = [("2026", "రెండు వేల ఇరవై ఆరు"), ("12వ", "పన్నెండవ"), ("10:45", "పది నలభై ఐదు")]

def strip_sp(toks):
    while toks and toks[-1] == "sp":
        toks = toks[:-1]
    return toks

def load_lexicon():
    lex = {}
    for l in open("lexicon/librispeech-lexicon.txt", encoding="utf8"):
        t = re.split(r"\s+", l.strip("\n"))
        lex.setdefault(t[0].lower(), t[1:])
    return lex

def en_tokens(text, lex, g2p):
    text = text.replace("’", "'")
    for a, b in EN_NORM:
        text = text.replace(a, b)
    out = []
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        lw = w.lower()
        if lw in EN_OVR:
            out += EN_OVR[lw]
        elif lw in lex:
            out += lex[lw]
        else:
            out += [p for p in g2p(w) if p != " "]
    s = "{" + "}{".join(out) + "}"
    s = re.sub(r"\{[^\w\s]?\}", "{sp}", s).replace("}{", " ")
    return strip_sp(s.strip("{}").split())
"""FastSpeech 2 inference for Telugu (run from this folder: python inference.py)."""
import os, re, sys, json, yaml, glob, zipfile, unicodedata, argparse
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.io import wavfile

LANG = "Telugu"
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
from text import text_to_sequence
from model import FastSpeech2
from utils.model import get_vocoder, vocoder_infer

device = "cuda" if torch.cuda.is_available() else "cpu"

EN = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}
TE = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}
SENTENCES = EN if LANG == "English" else TE
DEFAULT_SPEAKER = "spk0" if LANG == "English" else "female"

# ---------------- text frontend ----------------
EN_NORM = [("10:45 AM", "ten forty five AMX"),
           ("12 November 2026", "twelve November twenty twenty six")]
EN_OVR = {"amx": ["EY1", "EH1", "M"]}
TE_NORM = [("2026", "రెండు వేల ఇరవై ఆరు"), ("12వ", "పన్నెండవ"), ("10:45", "పది నలభై ఐదు")]

def strip_sp(toks):
    while toks and toks[-1] == "sp":
        toks = toks[:-1]
    return toks

def load_lexicon():
    lex = {}
    for l in open("lexicon/librispeech-lexicon.txt", encoding="utf8"):
        t = re.split(r"\s+", l.strip("\n"))
        lex.setdefault(t[0].lower(), t[1:])
    return lex

def en_tokens(text, lex, g2p):
    text = text.replace("’", "'")
    for a, b in EN_NORM:
        text = text.replace(a, b)
    out = []
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        lw = w.lower()
        if lw in EN_OVR:
            out += EN_OVR[lw]
        elif lw in lex:
            out += lex[lw]
        else:
            out += [p for p in g2p(w) if p != " "]
    s = "{" + "}{".join(out) + "}"
    s = re.sub(r"\{[^\w\s]?\}", "{sp}", s).replace("}{", " ")
    return strip_sp(s.strip("{}").split())

def te_tokens(text):
    text = unicodedata.normalize("NFC", text)
    for a, b in TE_NORM:
        text = text.replace(a, b)
    toks = []
    for ch in text:
        if 0x0C00 <= ord(ch) < 0x0C80:
            toks.append(ch)
        elif ch in ",.;:!?-":
            if toks and toks[-1] != "sp":
                toks.append("sp")
        elif ch.isspace():
            pass
        else:
            print("skipped char:", repr(ch))
    return strip_sp(toks)

def to_ids(tokens, cleaners):
    ids = text_to_sequence("{" + " ".join(tokens) + "}", cleaners)
    assert len(ids) == len(tokens), "token missing from vocabulary"
    return ids
def te_tokens(text):
    text = unicodedata.normalize("NFC", text)
    for a, b in TE_NORM:
        text = text.replace(a, b)
    toks = []
    for ch in text:
        if 0x0C00 <= ord(ch) < 0x0C80:
            toks.append(ch)
        elif ch in ",.;:!?-":
            if toks and toks[-1] != "sp":
                toks.append("sp")
        elif ch.isspace():
            pass
        else:
            print("skipped char:", repr(ch))
    return strip_sp(toks)

def to_ids(tokens, cleaners):
    ids = text_to_sequence("{" + " ".join(tokens) + "}", cleaners)
    assert len(ids) == len(tokens), "token missing from vocabulary"
    return ids

def tel_font():
    for f in fm.findSystemFonts():
        if "telugu" in f.lower():
            return fm.FontProperties(fname=f)
    return fm.FontProperties(family="DejaVu Sans")

LAT = fm.FontProperties(family="DejaVu Sans")
TEL = tel_font()

def plot(labels, vals, ylabel, title, path, kind):
    fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(labels)), 3.6))
    x = list(range(len(vals)))
    if kind == "bar":
        ax.bar(x, vals, color="#4C78A8")
    else:
        ax.plot(x, vals, marker="o", ms=3, color="#E4572E")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    for t in ax.get_xticklabels():
        t.set_fontproperties(LAT if t.get_text().isascii() else TEL)
        t.set_fontsize(7)
    ax.set_ylabel(ylabel, fontproperties=LAT)
    ax.set_title(title, fontproperties=LAT, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--speaker", default=DEFAULT_SPEAKER)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "Results", LANG))
    args = ap.parse_args()

    for z in glob.glob("hifigan/*.zip"):
        if not os.path.exists(z[:-4]):
            zipfile.ZipFile(z).extractall("hifigan")

    pc = yaml.safe_load(open("preprocess.yaml"))
    mc = yaml.safe_load(open("model.yaml"))
    ckpt = args.ckpt or sorted(glob.glob("checkpoints/*.pth.tar"))[-1]
    model = FastSpeech2(pc, mc).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    vocoder = get_vocoder(mc, device)

    cleaners = pc["preprocessing"]["text"]["text_cleaners"]
    pp = pc["path"]["preprocessed_path"]
    stats = json.load(open(f"{pp}/stats.json"))
    p_mean, p_std = stats["pitch"][2:4]
    e_mean, e_std = stats["energy"][2:4]
    speakers = json.load(open(f"{pp}/speakers.json"))
    spk_id = speakers.get(args.speaker, 0)
    sr = pc["preprocessing"]["audio"]["sampling_rate"]
    hop = pc["preprocessing"]["stft"]["hop_length"]

    if LANG == "English":
        from g2p_en import G2p
        lex, g2p = load_lexicon(), G2p()
        front = lambda t: en_tokens(t, lex, g2p)
    else:
        front = te_tokens

    R = args.out
    V = f"{R}/variance_adaptor_plots"
    for sub in ["test_generated_audios", "variance_adaptor_plots/duration",
                "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"]:
        os.makedirs(f"{R}/{sub}", exist_ok=True)

    for sid, text in SENTENCES.items():
        toks = front(text)
        ids = to_ids(toks, cleaners)
        with torch.no_grad():
            out = model(torch.tensor([spk_id], device=device),
                        torch.tensor([ids], device=device),
                        torch.tensor([len(ids)], device=device), len(ids))
            wav = vocoder_infer(out[1].transpose(1, 2), vocoder, mc, pc)[0]
        p = out[2][0].cpu().numpy() * p_std + p_mean
        e = out[3][0].cpu().numpy() * e_std + e_mean
        d = out[5][0].cpu().numpy() * hop / sr * 1000
        wavfile.write(f"{R}/test_generated_audios/{sid}.wav", sr, wav)
        plot(toks, d, "duration (ms)", f"{LANG} {sid}: duration", f"{V}/duration/{sid}.png", "bar")
        plot(toks, p, "F0 (Hz)", f"{LANG} {sid}: pitch", f"{V}/pitch/{sid}.png", "line")
        plot(toks, e, "energy", f"{LANG} {sid}: energy", f"{V}/energy/{sid}.png", "line")
        print(f"{LANG} {sid}: symbols={len(ids)} audio={len(wav)/sr:.2f}s")

if __name__ == "__main__":
    main()
