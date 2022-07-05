# LEGO (Learning Equality and Group Operation) Dataset
###      from paper [Unveiling Transformers with LEGO](https://arxiv.org/abs/2206.04301)

LEGO is a synthetic task that encapsulates the problem of following a chain of reasoning, proposed to study how the Transformer models learn and perform symbolic logics.
This initial release contains routines for generating the LEGO dataset and for reproducing an essential part of the experimental results in the paper.

## Requirements
Our current implementation is based on Pytorch and the Transformer library from Hugging Face. 

To install the required dependencies:

    pip install -r requirements.txt
    
## LEGO dataset generator
The code for generating the LEGO dataset is included in [lego_data.py](lego_data.py). One may obtain Pytorch dataloaders using the following call:

    trainloader, testloader = make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size)
    
## Training demo

We provide a demo of training Transformer model to solve LEGO in [demo.ipynb](demo.ipynb). A full run of 200 epochs may take ~3 hours on 4x Nvidia V100 GPUs.

## Citation
If you find our LEGO dataset useful, please cite our paper at

    @article{zhang2022unveiling,
    title={Unveiling Transformers with LEGO: a synthetic reasoning task},
    author={Zhang, Yi and Backurs, Arturs and Bubeck, S{\'e}bastien and Eldan, Ronen and Gunasekar, Suriya and Wagner, Tal},
    journal={arXiv preprint arXiv:2206.04301},
    year={2022}
    }
