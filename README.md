# CASCADE

**Context-Aware Significance of Cross-gene Attribution for Discovering Elements (CASCADE) for DNA Large Language Models uncovers motifs at cell-type-specific, tissue-specific, and gene-specific resolutions**

Ali Farghadan<sup>1</sup>, Robert J. Schmitz<sup>2</sup>, Scott A. Jackson<sup>1</sup>, Ethan Pickering<sup>1</sup>

<sup>1</sup>Department of Crop and Soil Sciences, University of Georgia, Athens, GA, USA
<sup>2</sup>Department of Genetics, University of Georgia, Athens, GA, USA

## Abstract

Gene expression is governed by regulatory DNA and their associated trans factors acting in specific cell types, yet the sequences underlying this control remain poorly mapped in plants. Genome-pretrained DNA language models provide a route to interrogate regulatory sequence directly, but their attributions have largely been interpreted using bulk or whole-tissue data, and standard attribution pipelines can preferentially highlight sequences downstream of the transcription start (TSS) site rather than promoter-associated signals. Here, we train a cell-type-resolved sequence-to-expression model from a single-cell soybean (*Glycine max*) atlas by coupling a soybean-adapted Genomic Pre-trained Network (GPN) to a shared sequence encoder with 66 cell-type-specific output heads. Across 38,339 protein-coding genes, the model achieves a mean per-cell-type, across-gene Pearson correlation of 0.683 and, recast as a high-versus-low expression classification, reaches an area under the ROC curve of 0.92 to 0.97 across tissues, at or above dedicated plant sequence models. We then introduce Context-Aware Significance of Cross-gene Attribution for Discovering Elements (CASCADE), a position-specific statistical framework for identifying model-derived candidate regulatory elements from *in silico* saturation mutagenesis. Relative to the pooled null used by TF-MoDISco, CASCADE shifts motif recovery from downstream of the transcription start site toward promoter sequence, with 77% of CASCADE-exclusive motifs, compared with 12% of TF-MoDISco-exclusive motifs, falling within the promoter. Applied across the atlas, CASCADE identifies approximately 1.39 million candidate elements spanning broadly active, tissue-restricted and cell-type-restricted classes. Together, these analyses establish a position-aware approach for extracting promoter-associated regulatory hypotheses from sequence models and generate a cell-type-resolved map of candidate *cis*-regulatory elements.

## What's here

This repository is being built out to accompany the paper. Currently it contains:

- **`train_example.py`** -- a minimal, single-file, illustrative training script for the cell-type-resolved sequence-to-expression (S2E) model described in the Methods. It trains a convolutional multi-task decoder on precomputed, frozen GPN embeddings to predict expression across all cell types at once. It is a simplified stand-in for the full training pipeline (no distributed training, cross-validation, or learning-rate scheduling), meant to make the core model and training procedure easy to read and reproduce on a small example.
- **`requirements.txt`** -- Python dependencies for `train_example.py`.

The single-cell soybean atlas, the per-gene attribution library, the full training/evaluation pipeline, and the CASCADE seqlet-calling code that reproduce the paper's figures will be added to this repository upon publication.

## Model summary

For each gene, a soybean-adapted GPN backbone (fine-tuned by masked-language modeling on *Glycine max* sequence, then frozen) embeds the TSS &plusmn; 2,000 bp window at single-nucleotide resolution, giving a fixed feature matrix `X_g in R^(L x D)` with `L = 4,000` and `D = 512`. A convolutional decoder (two 1-D convolutions with batch normalization and ReLU, dropout, global max pooling, layer normalization, a low-rank gene-latent bottleneck, and a two-layer MLP head) maps `X_g` to one predicted expression value per cell type, for all 66 cell types simultaneously ("multi-task" training). `train_example.py` implements this decoder and training loop; it expects precomputed embeddings as input rather than running the GPN backbone itself.

## Running the example

```bash
pip install -r requirements.txt

python train_example.py \
  --embeddings_file /path/to/embeddings.safetensors \
  --expression_path /path/to/expression.csv \
  --output_dir ./example_run
```

- `--embeddings_file` should hold a mapping `{gene_id: FloatTensor(L, D)}` for every gene (safetensors or `.pt`).
- `--expression_path` should be a CSV with genes as rows (first column = gene ID) and cell types as columns.

See `train_example.py --help` for the full set of training arguments (model width, dropout, learning rate, batch size, early-stopping patience, etc.).

## Data and code availability

The single-cell soybean atlas, the per-gene attribution library, and the code that trains the model and reproduces the figures will be made publicly available in this repository upon publication.

## Citation

A citation will be added here once the manuscript is published.

## License

Released under the MIT License (see `LICENSE`).
