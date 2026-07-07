# The Wall, the Game, and the Cliff Criterion

**Author:** Guangyu Wang (王广宇) · [billgywang@gmail.com](mailto:billgywang@gmail.com)
**GitHub:** [@gwang1688](https://github.com/gwang1688) · [project repository](https://github.com/gwang1688/The-Wall-the-Game-and-the-Cliff-Criterion)

A theory core and two applications on **sharp thresholds in monitoring and verification**, built on the
partial-channel / spiked-covariance model. Each paper is available in **English and Simplified Chinese**.
Every quantitative claim is verified by simulation; the scripts are in [`code/`](code).

---

## The three papers

### 1. Core — *The Wall, the Game, and the Cliff Criterion* · [`core/`](core)
The theory core. A planted signal is hidden through a partial channel; recovery has a sharp **wall**
`ρ★ = 1 − √(m/n)/s²` (BBP edge), certified by a matching Le Cam converse and a **triple coincidence** (wall =
diverging shadow price = signal energy crossing into the null channel). Includes the CGMT **game value**, the
extensive/saturation regimes, and the **Cliff Criterion**: a second-order footprint read by a committed
observer gives a cliff, while an averaged footprint or a forced-randomization (covering) observer gives a
**slope**. Dimensional analysis self-consistent; eleven review points incorporated.
Files: `core.{tex,pdf}` (EN, 21 pp), `core_zh.{tex,pdf}` (ZH, 19 pp).

### 2. Monitoring — *The Monitoring Wall and the Monitorability Tax* · [`monitoring/`](monitoring)
Chain-of-thought safety application of the core: when does a monitor reading a model's reasoning hit the
wall, and what is the tax of staying monitorable? Cites the core and is self-contained via its appendix.
Files: `monitoring_wall.{tex,pdf}` (EN, 19 pp), `monitoring_wall_zh.{tex,pdf}` (ZH, 17 pp).

### 3. Coverage — *The Coverage Wall and the Verification Tax* · [`coverage/`](coverage)
RLVR (reinforcement learning with verifiable rewards) application: the coverage wall for a verifier and its
verification tax, transplanted by an isomorphism lemma that decouples the application from the core proofs.
Stage-A verified. Files: `coverage_wall.{tex,pdf}` (EN, 9 pp), `coverage_wall_zh.{tex,pdf}` (ZH, 8 pp).

---

## Repository layout

```
.
├── core/          core paper (EN + ZH), .tex + .pdf + its figures
├── monitoring/    monitoring application (EN + ZH), .tex + .pdf + its figures
├── coverage/      coverage application (EN + ZH), .tex + .pdf + its figures
└── code/          numpy/scipy scripts that generate the figures and verify the claims
```

Each paper folder is **self-contained**: the `.tex`, the compiled `.pdf`, and the figures that paper needs
are co-located, so each compiles in place.

## Building the PDFs

The compiled PDFs are included. To rebuild:

- **English papers** (`core`, `monitoring_wall`, `coverage_wall`) — use **pdfLaTeX**:
  ```bash
  cd core && pdflatex core.tex && pdflatex core.tex
  ```
- **Chinese papers** (`*_zh`) — use **XeLaTeX** (they load `fontspec` with *Noto Serif CJK SC*):
  ```bash
  cd core && xelatex core_zh.tex && xelatex core_zh.tex
  ```
  Install the CJK font if needed (e.g. `fonts-noto-cjk` on Debian/Ubuntu).

Run LaTeX twice so cross-references and the table of contents resolve.

## Code

`code/` contains the CPU-only `numpy`/`scipy` scripts used to produce the figures and to verify the
theorems numerically (root-counts, CGMT checks, converse checks, effective-rank and two-timescale probes,
covering-slope checks, stage-A verification, etc.). No GPU or heavy dependencies are required:
```bash
pip install numpy scipy matplotlib
python code/tt_model.py
```

## Citation

```bibtex
@misc{wang_wall_cliff,
  author = {Guangyu Wang},
  title  = {The Wall, the Game, and the Cliff Criterion},
  year   = {2026},
  howpublished = {\url{https://github.com/gwang1688/The-Wall-the-Game-and-the-Cliff-Criterion}}
}
```

## Contact

Guangyu Wang (王广宇) — billgywang@gmail.com
