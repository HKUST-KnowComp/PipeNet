# Parse-LM-GNN

This is the code repo for *SEM 2024: PipeNet: Question Answering with Semantic Pruning over Knowledge Graphs

## Envs
   python: 3.7.13
   transformers: 3.4.0
   
## Data Preparation

   We use the question answering datasets ([CommonsenseQA](https://allenai.org/data/commonsenseqa) and [OpenBookQA](https://allenai.org/data/open-book-qa)), as well as ConceptNet as the knowledge source.
   ```
   sh download_raw_data.sh
   ```

   The datasets are processed by `preprocess_prune.py`. It grounds the concept in ConceptNet with calculating the distance score at the same time. It also generates pruned knowledge subgraph. e.g., preprocess CSQA with prune rate 0.9:
   ```
   python preprocess_prune.py --run csqa --prune 0.9
   ```

## Training
   For CommonsenseQA, run
   ```
   sh run_pipenet_csqa.sh
   ```

   For OpenBookQA, run
   ```
   sh run_pipenet_obqa.sh
   ```

   For OpenBookQA with additional fact knowledge, download the pretrained model [AristoRoBERTa](https://huggingface.co/LIAMF-USP/aristo-roberta/tree/main) first.

## Evaluation
   For CommonsenseQA, run
   ```
   run eval_pipenet_csqa.sh
   ```

   For OpenBookQA, run
   ```
   run eval_pipenet_obqa.sh
   ```
