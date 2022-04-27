# Intrinsic Prompt Tuning

Source code and dataset for the paper: "[Exploring Low-dimensional Intrinsic Task Subspace via Prompt Tuning](https://arxiv.org/abs/2110.07867)".

***
### Environment

```bash
conda create -n ipt python=3.6.9
conda activate ipt
pip install datasets==1.4.0 py7zr wget
pip install torch==1.8.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0 transformers==4.9.0 h5py==3.1.0 numpy==1.19.5
```
***
### Data

The data processing sctrictly follows [crossfit](https://github.com/INK-USC/CrossFit), we have already provided the data used in our paper: [few-shot_data_link](https://drive.google.com/file/d/1gooRoE81crfSa5iodzYCkQcGTJVTFEBu/view?usp=sharing) / [full-size_data_link](https://cloud.tsinghua.edu.cn/f/23dee716b51f45988c2f/?dl=1).


***
### Training & Evaluation 

#### Vanilla Prompt Tuning
```shell
bash example_scripts/promptTune.sh
```

We have provided all the trained prompts in [prompt_link](https://cloud.tsinghua.edu.cn/f/e85b9604def14f4e8455/?dl=1).

#### Multi-task Subspace Finding
To find a prompt subspace, run the following:
```shell
bash example_scripts/MSF_prompt.sh
```
We provide the trained autoencoders (the random task split) with different intrinsic dimension (3/5/10/50/100) at [link](https://cloud.tsinghua.edu.cn/d/73b7c3e3d6f945f597ea/).

#### Intrinsic Subspace Tuning
(1) Tuning intrinsic vectors in a prompt subsapce found by IPT, run the following: 
```shell
bash example_scripts/IST_prompt.sh
```

(2) Tuning intrinsic vectors in a random subspace for all parameters in the PLM defined by Fastfood Transformation, run the following:
```shell
bash example_scripts/Tune_said.sh
```

(3) Tuning intrinsic vectors in a random prompt subspace, use the script example_scripts/IST_prompt.sh with --AE_recover_random

(4) Tuning intrinsic vectors in an adapter subsapce found by IPT, run the following:
```shell
bash example_scripts/IST_adapter.sh
```

(5) For combining prompt tuning and IPT, i.e., using the solution found by IPT as the initialization for prompt tuning, and then conduct prompt tuning, run the following:
```shell
bash example_scripts/IST_stage_two.sh
```