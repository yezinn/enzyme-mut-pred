# Enzyme Variant Effect Prediction with ESM2

Zero-shot mutation effect prediction for industrial enzymes using the ESM2 protein language model, with a hands-on extension into structure prediction (Boltz-2). Built to demonstrate applied protein-AI methodology (protein language models, structure/stability prediction) for enzyme engineering.

## Motivation

Most of my prior ML/bioinformatics work (M.S. thesis on transfer-learning-based drug response prediction, published transcriptomic meta-analysis, whole-slide-image classification) is transcriptomics- and pathology-image-focused. This project was built specifically to gain hands-on experience with **protein sequence/structure foundation models** (ESM2, and Boltz-2 for structure), which is a distinct and increasingly important subfield within computational biology — directly relevant to AI-driven enzyme/protein engineering roles.

## Method

ESM2 (`facebook/esm2_t33_650M_UR50D`) is used to score the effect of point mutations on protein function **without any experimental data**, using the masked-marginal zero-shot scoring method from [Meier et al., 2021](https://doi.org/10.1101/2021.07.09.450648) ("Language models enable zero-shot prediction of the effects of mutations on protein function"):

1. Mask the mutated position in the wild-type sequence.
2. Get the model's predicted log-probability distribution over amino acids at that position.
3. Score = log P(mutant residue) − log P(wild-type residue). Higher score ⇒ the model considers the mutation more evolutionarily plausible (i.e., less likely to be deleterious).

**Performance optimization**: naively scoring one mutation at a time requires one forward pass per mutation (~5,000 forward passes for a full DMS dataset — 10+ hours on CPU). Since multiple mutations share the same masked position, this was refactored to cache one forward pass per *position* (batched), reducing compute by ~19x with no change in output.

## Phase 1 — Benchmark validation

Validated the pipeline against a real deep mutational scanning (DMS) dataset: TEM-1 β-lactamase (*E. coli*, Ranganathan lab 2015 — the same dataset used in the original ESM zero-shot paper).

**Result: Spearman ρ = 0.730** between ESM2 zero-shot scores and experimental fitness (ampicillin selection at 2500 μg/mL), matching the ESM1v 5-model ensemble performance reported in the original paper (ρ ≈ 0.7).

![Phase 1 correlation](phase1_blat_ecolx_correlation.png)

## Phase 2 — Application to industrial enzymes

Applied the validated pipeline to two enzymes relevant to industrial biotech (fermentation/feed enzyme production):

| Enzyme | UniProt | Application |
|---|---|---|
| Phytase A (*A. niger*) | [P34752](https://www.uniprot.org/uniprotkb/P34752/entry) | Feed enzyme — improves phosphorus bioavailability in animal feed |
| Alpha-amylase / amyS (*B. licheniformis*) | [P06278](https://www.uniprot.org/uniprotkb/P06278/entry) | Starch liquefaction — food/fermentation industry standard enzyme |

For each enzyme, ran full saturation mutagenesis (every position × 19 substitutions) and validated the model's biological plausibility with a sanity check: comparing predicted deleteriousness of mutations at known catalytic/active-site residues (from UniProt annotations) vs. binding-site vs. all other positions.

| Enzyme | Active site mean LLR | Binding site mean LLR | Other positions mean LLR |
|---|---|---|---|
| Phytase A | −11.94 (His82) | −8.92 | −5.29 |
| Alpha-amylase | −13.33 (Asp260, Glu290) | −6.17 | −5.24 |

Both enzymes show the expected gradient (active site mutations scored most deleterious), confirming the model correctly identifies functionally critical residues without being told which ones they are. For alpha-amylase, the identified active-site residues (Asp260/Glu290, in UniProt precursor numbering) correspond exactly to the literature-known catalytic triad (Asp-Glu-Asp) of GH13-family amylases.

Top-ranked candidate variants (highest predicted "fitness" mutations) are saved per enzyme as CSVs for downstream hypothesis generation.

## Phase 3 (in progress) — Structure prediction with Boltz-2

Extending the pipeline to compare predicted 3D structure/confidence of wild-type vs. top ESM2-ranked variants using [Boltz-2](https://github.com/jwohlwend/boltz) (open-source, MIT license), an AlphaFold3-class model that can also predict ligand binding affinity. See `boltz2_structure_extension.ipynb`.

## Limitations

- Zero-shot LLR is a proxy for evolutionary plausibility / stability, **not** a direct measurement of catalytic activity or substrate specificity. Top-ranked candidates are hypotheses for further (wet-lab or higher-fidelity computational) validation, not confirmed improvements.
- Structure prediction (Boltz-2), binding affinity prediction, and de novo homolog discovery are follow-on work, not yet fully executed as of this writing (as of 2026-09-17).

## Repo contents

- `esm2_enzyme_variant_effect_miniproject.ipynb` — Phase 1 + Phase 2 (main pipeline)
- `boltz2_structure_extension.ipynb` — Phase 3 (structure prediction extension, in progress)
- `BLAT_ECOLX_reference_with_esm1v_scores.csv` — reference DMS dataset + published ESM1v scores (validation)
- `phase2_*_saturation_mutagenesis.csv` — per-enzyme saturation mutagenesis results
- `phase1_blat_ecolx_correlation.png`, `phase2_heatmaps.png` — result figures

## References

- Meier et al., 2021. [Language models enable zero-shot prediction of the effects of mutations on protein function](https://doi.org/10.1101/2021.07.09.450648). bioRxiv.
- Lin et al., 2023. Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM2). *Science*.
- Passaro et al., 2025. [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://github.com/jwohlwend/boltz).
