#!/usr/bin/env python3
import argparse

BW2AR = {
    "'": "ء", ">": "أ", "<": "إ", "&": "ؤ", "}": "ئ", "|": "آ",
    "A": "ا", "b": "ب", "t": "ت", "v": "ث", "j": "ج", "H": "ح", "x": "خ",
    "d": "د", "*": "ذ", "r": "ر", "z": "ز", "s": "س", "$": "ش",
    "S": "ص", "D": "ض", "T": "ط", "Z": "ظ", "E": "ع", "g": "غ",
    "f": "ف", "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن",
    "h": "ه", "w": "و", "y": "ي", "Y": "ى", "p": "ة", "_": "ـ",
    "a": "َ", "u": "ُ", "i": "ِ", "F": "ً", "N": "ٌ", "K": "ٍ", "o": "ْ", "~": "ّ", "^": "ٰ",
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
            left, right = s.split("|", 1)
            fout.write(f"{left}|{bw2ar(right)}\n")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    convert(args.in, args.out)
