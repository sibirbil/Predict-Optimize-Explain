# Predict-Optimize-Explain

**Explaining portfolio optimization problems.**

Predict-Optimize-Explain (POE) explains trained predict-then-optimize (PTO) and predict-and-optimize (PAO) portfolio pipelines by reversing the usual question: which economies would drive a frozen pipeline toward a chosen decision-level behavior? Rather than attributing a single forecast, it generates a distribution of plausible macroeconomic states that produce a target behavior, keeping every generated scenario economically credible.

Each explanation is cast as a Gibbs distribution: a plausibility prior tilted by a probing loss on the frozen pipeline. The prior is a VAR(1) law fit to the macro series and localized to an anchor month. The target is sampled with Markov chain Monte Carlo, since its normalizing constant is intractable but its density ratio is not.

## Links

- Paper: Ataş, Aydın, Kıral & Birbil (2026), *Generating Input Distributions for Explaining Portfolio Optimization Pipelines*, [arXiv:2606.25808](https://arxiv.org/abs/2606.25808)
- Presentation: [POE pre-defense slides](https://sibirbil.github.io/files/poe/POE_presentation.html) (self-contained HTML, opens in any browser)

## Method

- **Pipelines.** A return predictor (neural network or gradient-boosted trees) feeds a robust mean–variance optimizer over a long-only simplex.
- **Probing functions.** Each explanation target is a loss on the frozen pipeline that is low exactly where the behavior holds: a benchmark-return probe, a concentration probe, and a PTO-vs-PAO divergence probe.
- **Plausibility prior.** A VAR(1) stationary law, tilted toward an anchor month, defines what counts as a plausible economy.
- **Samplers.** A MALA sampler for differentiable (neural) pipelines, and a gradient-free affine-invariant ensemble sampler for non-differentiable (tree) pipelines that expose no slope.

## Computational study
 
The pipelines are trained and probed on monthly U.S. equity data. Firm characteristics come from the Open Source Asset Pricing dataset (140 rank-normalized signals); the macro state is the nine Goyal-Welch predictors. The economy enters the forecast only through the 1,400 characteristic-by-macro interactions. Data are split into 192 training, 120 validation, and 107 test months, with the test window running 2016-2024.

## Getting started

```bash
git clone https://github.com/sibirbil/Predict-Optimize-Explain.git
cd Predict-Optimize-Explain
# install dependencies, then run an experiment from scripts/ or experiments/
```

## Authors

Batuhan Ataş, Nurşen Aydın, E. Mehmet Kıral, and Ş. İlker Birbil.

## Citation

```bibtex
@misc{ataş2026generatinginputdistributionsexplaining,
      title={Generating Input Distributions for Explaining Portfolio Optimization Pipelines}, 
      author={Batuhan Ataş and Nurşen Aydın and E. Mehmet Kıral and Ş. İlker Birbil},
      year={2026},
      eprint={2606.25808},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2606.25808}, 
}
```
