# CORVUS-CORAX Research

Official research publications, mathematical foundations, and verification scripts for the CORVUS-CORAX hybrid neural architecture framework.

## Overview

CORVUS-CORAX is a hybrid neural architecture combining:
- **Mixture of Experts (MoE)** with auxiliary-loss-free load balancing
- **Multi-Head Latent Attention (MLA)** with 8x KV-cache compression
- **Selective State Spaces (Mamba)** with O(n) theoretical complexity

This repository contains the academic research and reproducibility materials. The source code implementation is maintained separately.

## Repository Structure

```
corax-research/
├── papers/                 # Academic papers (CC BY-NC 4.0)
│   ├── corvus_mathematical_foundations.tex
│   ├── corvus_mathematical_foundations.pdf
│   └── references.bib
├── docs/                   # Technical documentation (CC BY 4.0)
│   ├── MATHEMATICAL_FOUNDATIONS.md
│   ├── VERIFICATION_SUMMARY.md
│   └── mathematical_verification_report.json
├── verification/           # Reproducibility scripts (Apache 2.0)
│   └── mathematical_verification.py
└── benchmarks/             # Benchmark suites (Apache 2.0)
```

## Quick Start

### Run Verification Suite

```bash
# Clone repository
git clone https://github.com/cuervo-ai/corax-research.git
cd corax-research

# Install dependencies
pip install torch numpy scipy

# Run verification
python verification/mathematical_verification.py --device auto --seed 42
```

### Expected Results

| Test | Expected | Status |
|------|----------|--------|
| MoE Load Balancing | Entropy > 0.8 | PASS |
| MLA Compression | >= 7x | PASS (8x) |
| MLA Attention O(n²) | R² > 0.95 | PASS |
| Numerical Stability | 0 NaN/Inf | PASS |

## Papers

### Mathematical Foundations (2026)

**Title:** CORVUS-CORAX: Mathematical Foundations and Empirical Verification of Hybrid Neural Architectures

**Abstract:** We present rigorous mathematical foundations for CORVUS-CORAX, verifying MLA compression (8x), MoE load balancing (entropy 0.9999), and attention complexity (O(n²), R²=0.9999).

**Download:** [PDF](papers/corvus_mathematical_foundations.pdf) | [LaTeX](papers/corvus_mathematical_foundations.tex)

## Citation

```bibtex
@techreport{corvus2026mathematical,
  title={CORVUS-CORAX: Mathematical Foundations and Empirical Verification
         of Hybrid Neural Architectures},
  author={{CORVUS-CORAX Research Team}},
  year={2026},
  month={January},
  institution={Cuervo AI},
  url={https://github.com/cuervo-ai/corax-research}
}
```

## Licensing

This repository uses a multi-license structure:

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Papers | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | No |
| Documentation | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes |
| Scripts | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes |

See [LICENSE](LICENSE) for full details.

## Source Code

The full implementation of CORVUS-CORAX is maintained in a separate repository under different licensing terms.

For commercial licensing inquiries: contact@cuervo.ai

## References

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
3. DeepSeek-AI (2024). "DeepSeek-V3 Technical Report"
4. Kapoor et al. (2024). "REFORMS: Recommendations for ML-based Science"

## Contributing

We welcome contributions to:
- Verification scripts and benchmarks
- Documentation improvements
- Bug reports for verification code

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Maintained by:** Cuervo AI
**Contact:** contact@cuervo.ai
**Website:** https://cuervo.ai
