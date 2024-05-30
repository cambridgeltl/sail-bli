# SAIL-BLI
This repository is the official PyTorch implementation of the following paper:

Yaoyiran Li, Anna Korhonen, and Ivan Vulić. 2024. *Self-Augmented In-Context Learning for Unsupervised Word Translation*. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024). [[Paper]](https://arxiv.org/abs/2402.10024)

**SAIL** aims to improve **unsupervised** BLI by **(1)** inferring a high-confidence word translation dictionary with zero-shot prompting, **(2)** then optionally refining the high-confidence dictionary iteratively with few-shot prompting where the in-context examples are from the high-confidence dictionary in the previous iteration, and **(3)** finally conducting evaluation on the BLI test set with few-shot prompting also deriving in-context samples from the latest high-confidence dictionary. The whole process does not leverage any ground-truth word translation pairs for training/few-shot learning and improves the BLI scores by typically 10 ~ 15 P@1 points comparing to zero-shot prompting.

# Dependencies
- PyTorch>=1.10.1
- Transformers>=4.28.1

# Data
Following [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/), [BLICEr](https://github.com/cambridgeltl/BLICEr) and [Prompt4BLI](https://github.com/cambridgeltl/prompt4bli), our data is obtained from the [XLING](https://github.com/codogogo/xling-eval) (8 languages, 56 BLI directions in total) and [PanLex-BLI](https://github.com/cambridgeltl/panlex-bli) (15 lower-resource languages, 210 BLI directions in total).

# Run the Code
Prepare BLI Vocabulary:
```bash
python run_extract_vocabularies.py
```

Run BLI Evaluation (define key hyper-parameters, directories and language pairs to evaluate manually in run_bli.py):
```bash
python run_bli.py
```

# Citation
Please cite our paper if you find **SAIL-BLI** useful.
```bibtex
@inproceedings{li-etal-2024-self,
    title     = {Self-Augmented In-Context Learning for Unsupervised Word Translation},
    author    = {Li, Yaoyiran and Korhonen, Anna and Vuli{\'c}, Ivan},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},    
    year      = {2024}
}
```
