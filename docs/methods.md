# Models and evidence fusion

## Image model

Frozen patch embeddings from all slides belonging to a patient form one bag.
A learned dense projection, layer normalization, GELU and dropout transform the
patch vectors. Gated attention scores each valid patch using tanh and sigmoid
branches; padding receives exactly zero weight. A weighted sum creates the
patient representation, followed by a fusion layer and a classification head.
The default dimensions are 768 → 256, attention width 128, and fusion width 128.

Weighted cross-entropy uses training-only inverse-frequency weights. The batch
loss is the arithmetic mean of patient losses. Optional label smoothing, focal
weighting, AdamW and tempered inverse-frequency weights are explicit presets.
CONCH itself is frozen and is not loaded or trained by this repository.

## Hierarchical variants

**Joint hard hierarchy (B).** The original three groups are
`{C1,C2,C3}`, `{C4,C6,C8}`, and `{C5,C7,C9,C10}`. A shared patch projection feeds
independent root/child attention branches. Root and child losses are learned
together. Each patient's child loss uses its true group during training, even
when the root prediction is wrong. Inference chooses the root first and masks
all other groups.

**Joint global head (C).** A global ten-class head consumes all branch features
and all branch probability vectors. Its loss is added to the root and true-child
losses. The global head can select a class from any group; root probabilities
provide learned context rather than a hard mask. Historical B/C/D presets allow
the original learned direct age/location prior inside the network.

**Candidate epoch selection (D).** The C architecture is evaluated on inner
validation at epochs 5, 10, 15, 20 and 25. Selection maximizes correct count, then
minimizes unweighted NLL. A fresh model is then trained on train + validation
for the selected number of epochs. Outer test never selects the epoch. This
differs from conventional early stopping, which restores an existing checkpoint
and does not refit.

**Strict six-to-ten hierarchy.** First train a six-group image Teacher. Then
train independent image Teachers for G1 and G2 using only training patients
whose true class belongs to that parent. Singleton groups need no child. At
prediction time choose one parent and, when needed, one class inside that parent.
The child cannot repair a parent routing error. The default workflow therefore
uses three supervised image Teachers per fold, trained serially, with no Student
or ensemble; the four evidence settings do not train four copies of these models.

## When evidence is used

Let `p` be the image probabilities after temperature scaling. Evidence is
eligible when `max(p) < 0.8` **or** the difference between the two largest
probabilities is less than `0.2`. The same image-based gate is used for all four
evidence modes; evidence does not recursively redefine uncertainty. Confidently
wrong image predictions can therefore remain wrong.

Temperature is chosen by inner-validation NLL. Evidence strengths include zero
and are selected by validation correct count, NLL, then smaller strength.
Selection happens before any outer-test feature inference.

## Stage-one age/location cues

Five predefined cue families are retained from the original evidence code:

| Observed information | Candidate group boost before strength scaling |
|---|---|
| Brainstem/tectum or thalamus/deep gray | G1: +1 |
| Pineal region | G2 and G5: +1 each |
| Sellar/suprasellar/optic region | G1 and G5: +1 each |
| Ventricular/periventricular site | G3: +0.5 |
| Ventricular site with observed age ≤3 years | G4: +1 in addition to the site cue |

Only groups already among the image's top two candidates receive these boosts.
Missing location disables all five cues. The age cue also requires observed age.
The numerical boosts, group mappings and cutoffs are experimental heuristics,
not validated diagnostic rules. Their clinical motivation is documented in the
[references](references.md).

## IHC likelihoods

For each marker, count observed POS/NEG calls in the training partition. First
estimate a smoothed pooled positive rate. Shrink each class/group's positive
rate toward that pooled rate with eight pseudo-observations. An observed POS or
NEG contributes its corresponding log-likelihood ratio relative to the pool.

Each marker's contribution is clipped to ±log(3). The sum is divided by the
square root of the number of observed markers and clipped to ±log(4). This
tempers correlated markers. Missing observations add no score. Root statistics
use all training patients; child statistics use only that parent's training
patients. Held-out patients never estimate these rates.

At the root the final scores are:

```text
adjusted_logits = image_logits / temperature
                  + alpha * clinical_cue_scores
                  + beta  * ihc_log_likelihood_scores
```

Softmax converts the scores to probabilities. With no eligible evidence or
zero selected strength, the original image probabilities are preserved exactly.

## Stage-two clinical evidence

The newer child refinement keeps the root route, child image network and IHC
settings fixed. Age bins are ≤2, >2 to ≤10, and >10 years. Categorical
likelihoods are fitted using only child training patients and shrunk toward the
parent distribution. Eligible sites are:

- G1: cerebellum, brainstem/tectum, thalamus/deep gray, sellar/suprasellar/optic.
- G2: cerebellum, posterior fossa NOS, pineal region.

An age bin/site requires at least 5/3 matching training observations and a
1.5-fold likelihood contrast across the three child classes. Scores are clipped
to ±log(2) for age and ±log(3) for site; their weighted sum is clipped to ±log(4).
Age weights are 0/0.5/1 and site weights 0/0.5/1/1.5. Only true-parent inner
validation selects them. All three child candidates remain eligible, including
the image's third-ranked class.

## Reporting

Out-of-fold predictions cover each patient once. Accuracy is patient weighted;
balanced accuracy averages recall over supported classes; macro F1 averages
over the complete fixed label set. Final hard probabilities are zero outside
the chosen parent. A wrong route therefore produces infinite ten-class NLL,
stored as `null` together with `nll_status: infinite` and a zero-probability count.
Six-group routing scores are reported separately from final ten-class scores.
