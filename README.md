# Exponential-Family Membership Inference — code demo

Anonymous code accompanying a paper under review. This is a self-contained
NumPy implementation of the attack scoring functions (LiRA, RMIA, the BASE1–4
hierarchy, and BaVarIA-$n$ / BaVarIA-$t$) plus a minimal data bundle and a
notebook that reproduces two ROC figures end-to-end and verifies element-wise
agreement with the original pipeline on the same bundle.

Minimal Location/MLP-3 sample (5,010 samples, 5 antithetic shadow-pairs,
$K = 8$ max). About one minute on a laptop CPU. No GPU needed.

```
.
├── README.md
├── requirements.txt
├── methods.py            # LiRA, BASE1-4, RMIA, BaVarIA-n, BaVarIA-t (numpy)
├── utils.py              # bundle loader, antithetic-pair sampler, metrics
├── demo.ipynb            # produces the two figures + verification cell
└── data/                 # ~420 KB total
    ├── shadow_logits.npy        # (5010, 10) float32
    ├── shadow_mask.npy          # (5010, 10) bool
    ├── target_logits.npy        # (5010,)    float32
    ├── target_mask.npy          # (5010,)    bool
    ├── reference_scores_K8.npz  # scores from the original pipeline
    └── meta.json
```

| Path | Contents |
|---|---|
| `methods.py` | NumPy implementations of LiRA, BASE1-4, RMIA, BaVarIA-$n$, BaVarIA-$t$. |
| `utils.py` | Bundle loader, antithetic-pair sampler, ROC / AUC / TPR@FPR metrics. |
| `demo.ipynb` | End-to-end notebook: loads the bundle, runs all attacks, plots two ROC figures, and verifies element-wise agreement with `data/reference_scores_K8.npz`. |
| `data/shadow_logits.npy`, `shadow_mask.npy` | 10 shadow models = 5 antithetic pairs (columns $2i$ and $2i+1$ have complementary masks). |
| `data/target_logits.npy`, `target_mask.npy` | Held-out audit target. |
| `data/reference_scores_K8.npz` | Scores from the original pipeline on this exact bundle, for the verification cell. |
| `data/meta.json` | Bundle metadata. |
| `requirements.txt` | numpy, scipy, scikit-learn, matplotlib, jupyter. |

## Running

```bash
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

Run all cells. The notebook writes the two figures to `figures/` and prints an
`OK` on every line of the final verification cell when the demo methods agree
with the reference scores to float32 round-off.
