# Contamination-free replication: frozen WavLM and MiniLM with linear probes

## Table K-1 - Hyperparameters, selected once on MELD dev

| Probe               | Selected             | MELD dev bal acc |
|---------------------|----------------------|------------------|
| voice (WavLM-large) | layer L18, C = 0.001 | 57.36            |
| text (MiniLM-L6)    | C = 0.1              | 53.74            |

Searched: every stored WavLM layer x C in [0.001, 0.01, 0.1] for voice; C in [0.1, 1.0, 10.0] for text. These choices are then frozen for every corpus.

## Table K-2 - Voice across the ladder: clean probe beside deployed zero-shot

| Condition                      | n    | clean bal acc | clean macro F1 | deployed bal acc | deployed - clean |
|--------------------------------|------|---------------|----------------|------------------|------------------|
| CREMA-D (actor-grouped 5-fold) | 4900 | 86.94         | 86.77          | 93.28            | +6.34            |
| IEMOCAP script (LOSO)          | 2588 | 64.04         | 62.58          | 82.23            | +18.19           |
| IEMOCAP impro (LOSO)           | 2943 | 76.22         | 74.81          | 83.63            | +7.41            |
| IEMOCAP all (LOSO)             | 5531 | 72.01         | 70.68          | 83.76            | +11.74           |
| MELD test (train -> test)      | 2211 | 54.02         | 48.62          | 46.25            | -7.77            |

`deployed - clean` is NOT a contamination estimate by itself: the deployed model is zero-shot while the probe trains in-domain, which favours the probe, and emotion2vec+ is a stronger audio model than a linear probe on WavLM, which favours the deployed model. What survives contamination is the ORDERING of the rungs under the clean encoder.

## Table K-3 - Voice against text, clean encoders

| Corpus         | n    | clean voice bal acc | clean text bal acc | clean voice - text [95 % CI] | deployed voice - text |
|----------------|------|---------------------|--------------------|------------------------------|-----------------------|
| IEMOCAP (LOSO) | 5531 | 72.01               | 58.96              | +13.05 [+9.45, +16.17]       | +45.06                |
| MELD test      | 2211 | 54.02               | 53.99              | +0.03 [-3.19, +2.83]         | -5.59                 |

CIs are dialogue-clustered paired bootstraps. If the inversion were an artifact of each deployed encoder having memorised a different corpus, the clean margins would not change sign between the corpora.

## Table K-4 - Script against improvised within IEMOCAP, clean

| Modality | script bal acc | impro bal acc | script - impro [95 % CI] |
|----------|----------------|---------------|--------------------------|
| voice    | 64.04          | 76.22         | -12.18 [-16.71, -7.22]   |
| text     | 63.64          | 50.46         | +13.18 [+9.54, +16.52]   |

Dialogue-clustered bootstrap, script and improvised dialogues resampled independently.

## Table K-5 - Whole pipeline trained on one corpus, run on the other (clean, balanced accuracy)

| Component | MELD: in-domain | MELD: from IEMOCAP | MELD: from both | IEMOCAP: in-domain | IEMOCAP: from MELD | IEMOCAP: from both |
|-----------|-----------------|--------------------|-----------------|--------------------|--------------------|--------------------|
| voice     | 54.02           | 43.73              | 52.81           | 72.01              | 55.74              | 69.66              |
| text      | 53.99           | 40.32              | 52.70           | 58.96              | 38.73              | 55.29              |
| fusion    | 51.07           | 44.00              | 52.79           | 72.94              | 40.55              | 70.20              |

Cross-corpus minus in-domain, dialogue-clustered paired bootstrap:
- voice: on MELD -10.29 [-13.43, -7.52]; on IEMOCAP -16.27 [-18.74, -13.88]
- text: on MELD -13.68 [-16.83, -10.53]; on IEMOCAP -20.23 [-22.52, -17.92]
- fusion: on MELD -7.07 [-10.05, -3.90]; on IEMOCAP -32.39 [-34.27, -30.36]

## Table K-6 - WavLM layer sensitivity (voice probe, C fixed)

| Layer | IEMOCAP LOSO bal acc | MELD test bal acc | IEMOCAP - MELD |
|-------|----------------------|-------------------|----------------|
| L3    | 64.09                | 45.82             | +18.27         |
| L6    | 68.61                | 48.60             | +20.01         |
| L9    | 70.53                | 49.59             | +20.94         |
| L12   | 71.13                | 52.83             | +18.31         |
| L15   | 71.72                | 51.95             | +19.77         |
| L18   | 72.01                | 54.02             | +17.99         |
| L21   | 71.90                | 54.29             | +17.61         |
| L24   | 70.07                | 51.31             | +18.76         |
| mean  | 72.07                | 54.22             | +17.85         |

If the clean-against-wild gap held only at the selected layer it would be a selection artifact; it should hold across layers.

