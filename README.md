# LEGO (Learning Equality and Group Operation) Dataset
###      from paper [Unveiling Transformers with LEGO](https://arxiv.org/abs/2206.04301)

LEGO is a synthetic reasoning task that encapsulates the problem of following a chain of reasoning, proposed to study how the Transformer models learn and perform logical reaoning.
This initial release contains routines for generating the LEGO dataset and for reproducing an essential part of the experimental results in the paper.

## Requirements
Our current implementation is based on Pytorch and the Transformer library from Huggingface. 

To install the required dependencies:

    pip install -r requirements.txt
    
## Training DEMO

We provide a demo of training Transformer model to solve LEGO in [demo.ipynb](demo.ipynb). A full run of 200 epochs may take ~3 hours on 4x Nvidia V100 GPUs.

## Citation
If you find our LEGO dataset useful, please cite our paper at

    @article{zhang2022unveiling,
    title={Unveiling Transformers with LEGO: a synthetic reasoning task},
    author={Zhang, Yi and Backurs, Arturs and Bubeck, S{\'e}bastien and Eldan, Ronen and Gunasekar, Suriya and Wagner, Tal},
    journal={arXiv preprint arXiv:2206.04301},
    year={2022}
    }
