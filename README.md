## Training Trajectories of Language Models Across Scales

This repository contains the code and analysis results for our ACL'23 paper [Training Trajectories of Language Models Across Scales](https://arxiv.org/pdf/2212.09803.pdf).  

**************************** **Updates** ****************************
* 05/25/2022: Our paper gets accepted to ACL 2023! We update the camera ready version on ArXiv and release the code base.
* 12/19/2022: We released [our paper](https://arxiv.org/pdf/2212.09803.pdf). Check it out!

## Quick Links

- [Training Trajectories of Language Models Across Scales](#training-trajectories-of-language-models-across-scales)
- [Quick Links](#quick-links)
- [Overview](#overview)
- [Analysis](#analysis)
  - [Token-level analysis](#token-level-analysis)
  - [Sequence-level analysis for generated sequences](#sequence-level-analysis-for-generated-sequences)
  - [Downstream tasks](#downstream-tasks)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview

The objective of this study is to analyze the training process of large language models, specifically those with up to 175 billion parameters. The study focuses on examining the trends of token-level perplexity, sequence-level perplexity of generated sequences, and downstream task performance throughout the training duration. Our findings reveal that models of various scales, when pre-trained with identical data in the same sequence, exhibit similar behaviors and demonstrate comparable patterns as training progresses. This similarity in behavior is observed not at the same number of floating-point operations (FLOPs) or the same quantity of tokens processed, but rather at the same level of perplexity.

To illustrate this, the validation curves presented below depict five checkpoints (gray points) corresponding to different model sizes at a validation perplexity of 15. Despite their varying sizes, these checkpoints exhibit similar perplexity levels and yield comparable predictions, highlighting the consistency of model performance regardless of scale.

<img src="images/validation_ppl_annotated.png" data-canonical-src="images/validation_ppl_annotated.png" width="400"/>

## Analysis

Each part's analysis code is released in a separate notebook. Please note that the code for performing inference with model checkpoints, either for language modeling or downstream tasks, is not included in this repository. We solely provide code for the analysis conducted once all results have been gathered.

### Token-level analysis
This analysis consists of three parts:

1) The first part involves collecting token-level perplexity of model checkpoints on a document and saving it in the following format: {model_size: np.array(num_tokens, num_checkpoints)}. An example can be found in `data/all_ppls-gutenberg_pg-19.pt`.
2) The second part focuses on analyzing the trend of each token in intermediate checkpoints using the function `collect_trend_of_tokens` in `utils.py`.
3) The third part includes calculating the percentage of tokens exhibiting different trends (stagnated, upward trend, and downward trend). Additionally, it involves plotting the perplexity of tokens for each trend against FLOPs and validation perplexity. This analysis is presented in `analysis_section1.ipynb`.

Selected analysis: During the course of training, it was discovered that approximately **9.4%** of the tokens exhibited **a double-descent trend** in perplexity, as shown in the graph below.

<img src="images/section1/Increases-opt_1.3b-ppl-gutenberg_pg-19.png" data-canonical-src="images/section1/Increases-opt_1.3b-ppl-gutenberg_pg-19.png" width="400"/>

### Sequence-level analysis for generated sequences
The analysis comprises three main components:

1) In the first part, sequences are decoded by taking a weighted sum of predicted probabilities for the next token from two models of varying sizes. This functionality can be found in the utils.py file under the decode function.
2) The second part involves calculating the trajectories of the generated sequences and structuring the results in the format specified in data/generation_analysis/final_model_results.csv.
3) Finally, the perplexity of the generated sequences is plotted using the analysis_section2.ipynb notebook.

**Selected analysis**: We show that by subtracting the small model's probabilities from the large model's probabilities, greedy decoding could achieve similar generation quality as neucleus sampling. 


> Greedy decoding with $p_l - p_s$: </s>*The weather is nice today. Let's* go to the park and play some games!</s>  
> Nucleus sampling with $p_l - p_s$: </s>*The weather is nice today. Let's* go to the park for some sun and picture taking!</s> 

**Selected analysis**: We show that for inputs injected with noise and factually incorrect sequences, the perplexity still decreases as model size increases.

<img src="images/section2/noise.png" data-canonical-src="images/section2/noise.png" width="400"/>

**Selected analysis**: We found sequences that follow an inverse scaling trend ($p_s - p_l$) as the model size increases by maximizing the probablity of the small model and minimizing the probability of the large model. The following graph shows that such a inverse scaling trend holds across model families, where the sequences are generated with OPT models and are evaluated on the GPT-Neo models. 

<img src="images/section2/gpt-neo.png" data-canonical-src="images/section2/gpt-neo.png" width="400"/>

### Downstream tasks
This analysis consists of three parts: TO BE DONE

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Mengzhou (mengzhou@princeton.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you find our analysis helpful in your work:

```bibtex
@inproceedings{xia2023training,
   title={Training Trajectories of Language Models Across Scales},
   author={Xia, Mengzhou and Artetxe, Mikel and Zhou, Chunting and Lin, Xi Victoria and Pasunuru, Ramakanth and Chen, Danqi and Zettlemoyer, Luke and Stoyanov, Ves},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2023}
}
```
