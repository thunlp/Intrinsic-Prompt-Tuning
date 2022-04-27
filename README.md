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

The data processing sctrictly follows [crossfit](https://github.com/INK-USC/CrossFit), we have already provided the data used in our paper [data_link](https://drive.google.com/file/d/1gooRoE81crfSa5iodzYCkQcGTJVTFEBu/view?usp=sharing).


***
### Training & Evaluation

(1) For prompt tuning baseline, refer to example_scripts/promptTune_classification_singletasks_0.sh

(2) For multi-task subspace finding, refer to example_scripts/upstream_multitask_prompt_AE.sh

(3) For intrinsic subspace tuning, refer to example_scripts/upstream_multitask_prompt_AE_recover.sh

(4) For combining prompt tuning and IPT, refer to example_scripts/upstream_multitask_prompt_AE_recover_stage2.sh
