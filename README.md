# WiMamba-MoE: Mamba-Augmented Wireless Foundation Models

**MSc Thesis — BRAC University, CSE**

Mamba-Augmented Wireless Foundation Models for Semantic Communication in IoT-Dense 6G Networks.

## Smoke Test

The `smoke_test/` directory contains a minimal validation script designed to run on
Lightning AI with an A100 GPU. It uses **~1% of a single real DeepMIMO scenario**
(no synthetic data) to verify the full pipeline end-to-end before committing to
expensive multi-hour training runs.

### What the smoke test validates

1. CUDA / A100 GPU detection and memory reporting
2. DeepMIMOv3 dataset download (1 scenario only: `city_18_denver`)
3. Data slicing to ~1% of the scenario (first 500 user samples out of ~50,000+)
4. LWM (Large Wireless Model) pre-trained checkpoint loading from HuggingFace
5. Tokenization of real channel data into (N, 32, 32) patches
6. Forward pass through LWM Transformer encoder on GPU
7. Embedding extraction (CLS + channel embeddings)
8. NMSE baseline metric computation on masked channel reconstruction
9. Downstream task head smoke test (beam prediction with random labels)
10. Memory usage and timing report

### Running on Lightning AI

```bash
cd smoke_test
pip install -r requirements.txt
python run_smoke_test.py
```

Expected runtime: **< 5 minutes** on A100. Expected cost: **< $0.15**.

## Project Structure

```
├── README.md
├── smoke_test/
│   ├── requirements.txt
│   ├── run_smoke_test.py
│   └── config.py
├── src/                    # (future) main training code
├── configs/                # (future) experiment configs
└── results/                # (future) experiment outputs
```

## License

Research use only. See individual dataset and model licenses.
