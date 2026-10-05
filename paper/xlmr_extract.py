#!/usr/bin/env python3
"""XLM-RoBERTa-large text features, for the text-encoder-strength check.

Every voice-versus-text result in paper/ pairs a WavLM-large voice probe (~316M parameters)
with a MiniLM-L6 text probe (~22M). A reviewer can fairly object that text loses because
its encoder is smaller. This extracts features from xlm-roberta-large (~560M parameters,
LARGER than the voice encoder), so text_encoder_check.py can re-run every margin with it.

xlm-roberta-large was pretrained by masked language modelling on CommonCrawl text only, so,
like WavLM and MiniLM, it has seen no emotion labels. Web text may contain some of these
corpora's dialogue (for example television scripts), which is lexical exposure without
labels, the same status as WavLM's or emotion2vec's self-supervised audio exposure.

Stored per utterance: attention-masked mean of the hidden states at layers 6, 12, 16, 20,
24 and the mean over all 25, float16. The layer is chosen downstream on MELD dev, exactly
as WavLM's was. Rows align with paper/clean_feats/<group>.meta.jsonl.

Texts are the same ones the MiniLM features used: gold transcripts from the local-only
paper/.text_backup/ (IEMOCAP, MELD) and data/mosei/manifest.jsonl (MOSEI).

    .venv/bin/python paper/xlmr_extract.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

FEATS = "paper/clean_feats"
LAYERS = [6, 12, 16, 20, 24]
GROUPS = ["iemocap_all", "meld_train", "meld_dev", "meld_test", "mosei"]


def texts_for(group: str) -> list[str]:
    meta = [json.loads(l) for l in open(f"{FEATS}/{group}.meta.jsonl", encoding="utf-8")]
    if group == "mosei":
        by_id = {}
        for line in open("data/mosei/manifest.jsonl", encoding="utf-8"):
            r = json.loads(line)
            by_id[r["id"]] = r["text"]
        return [by_id[m["id"]] for m in meta]
    from clean_extract_features import manifest
    rows = manifest()[group]
    if [r["clip"] for r in rows] != [m["clip"] for m in meta]:
        sys.exit(f"! {group}: manifest order differs from clean_feats meta")
    return [r["text"] for r in rows]


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("xlm-roberta-large", local_files_only=True)
    model = AutoModel.from_pretrained("xlm-roberta-large", local_files_only=True).to("cuda").eval()
    for g in GROUPS:
        out = f"{FEATS}/{g}.xlmr.npz"
        if os.path.exists(out):
            print(f"{g}: exists")
            continue
        texts = texts_for(g)
        feats = np.zeros((len(texts), len(LAYERS) + 1, 1024), dtype=np.float16)
        order = np.argsort([len(t) for t in texts])          # length-sorted batches: less padding
        for b0 in range(0, len(texts), 64):
            idx = order[b0:b0 + 64]
            enc = tok([texts[i] if texts[i].strip() else "." for i in idx], padding=True,
                      truncation=True, max_length=256, return_tensors="pt").to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                hs = model(**enc, output_hidden_states=True).hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = torch.stack([(h.float() * mask).sum(1) / mask.sum(1) for h in hs], 1)
            sel = torch.cat([pooled[:, LAYERS], pooled.mean(1, keepdim=True)], 1)
            feats[idx] = sel.cpu().numpy().astype(np.float16)
        np.savez(out, xlmr=feats)
        print(f"{g}: {len(texts)} texts -> {out}", flush=True)


if __name__ == "__main__":
    main()
