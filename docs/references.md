# Method references

- Ilse, Tomczak and Welling. **Attention-based Deep Multiple Instance Learning**
  (2018). [Paper](https://proceedings.mlr.press/v80/ilse18a.html).
  Motivation for gated-attention pooling over variable-size patch bags.
- Mahmood Lab. **CONCH: a vision-language pathology foundation model**.
  [Official implementation and publications](https://github.com/mahmoodlab/CONCH).
  Embeddings are computed separately; encoder weights are not redistributed.
- Wang, Cuzzolin, Moens and Hallez. **Credal Ensemble Distillation for Uncertainty
  Quantification**. [Paper](https://arxiv.org/abs/2511.13766).
  Background for the historical CREDIT Student and CED loss modules.

The original clinical-cue review referred to the following sources:

- [Wong et al. (2005)](https://acsjournals.onlinelibrary.wiley.com/doi/full/10.1002/cncr.21430):
  broad pediatric tumor age/site context.
- [Wolff et al. / CPT-SIOP-2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC8860833/):
  context for young patients and choroid plexus tumors.
- [Pediatric pineal-region series (2023)](https://pubmed.ncbi.nlm.nih.gov/37148224/):
  context for the pineal-region differential.
- [NCI childhood glioma and glioneuronal tumor overview](https://www.cancer.gov/types/brain/hp/child-astrocytoma-glioma-treatment-pdq):
  broad clinical/pathological background for child-stage eligibility.

These sources motivate consideration of evidence. They do not establish the
project's six-group mapping, numerical boosts, confidence thresholds, age bins,
or likelihood caps as diagnostic rules. Those choices are research hypotheses
whose parameters and limitations are explicit in the code.
