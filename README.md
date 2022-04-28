# Intrinsic Prompt Tuning

Source code and dataset for the paper: "[Exploring Low-dimensional Intrinsic Task Subspace via Prompt Tuning](https://arxiv.org/abs/2110.07867)".

If you have any question, feel free to contact me by email (yujiaqin16@gmail.com).

***
## Environment

```bash
conda create -n ipt python=3.6.9
conda activate ipt
pip install datasets==1.4.0 py7zr wget
pip install torch==1.8.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0 transformers==4.9.0 h5py==3.1.0 numpy==1.19.5
```
***
## Data

The data processing sctrictly follows [crossfit](https://github.com/INK-USC/CrossFit), we have already provided the data used in our paper: [few-shot_data_link](https://drive.google.com/file/d/1gooRoE81crfSa5iodzYCkQcGTJVTFEBu/view?usp=sharing) / [full-size_data_link](https://cloud.tsinghua.edu.cn/f/23dee716b51f45988c2f/?dl=1).

We use three tasks splits in our paper: random, cls and non-cls. You could find the corresponding train / test task splits in dataloader/custom_tasks_splits/ + random.json/cls.json/noncls.json

***
## Training & Evaluation 

### Vanilla Prompt Tuning
```shell
bash example_scripts/promptTune.sh
```

We have provided all the trained prompts in [prompt_link](https://cloud.tsinghua.edu.cn/f/e85b9604def14f4e8455/?dl=1).

### Multi-task Subspace Finding
To find a prompt subspace, run the following:
```shell
bash example_scripts/MSF_prompt.sh
```
We provide the trained autoencoders with different intrinsic dimension at [link](https://cloud.tsinghua.edu.cn/d/73b7c3e3d6f945f597ea/). The autoencoders are trained on ~100 training tasks of the random task split. Please refer to dataloader/custom_tasks_splits/random.json to see how the tasks are divided into train/test tasks. The decoder in the autoencoder defines the subspace we find, which would be tested on the test tasks in IST.

The trained adapter intrinsic encoders are hosted at Ali-Cloud. Visit the following link: https://qinyujia.oss-cn-qingdao.aliyuncs.com/adapter_distil_bs256_dim5_type1_20w_random/best-ckpt.pt, change the number after dimension (e.g., 5, 10, 50) to access different checkpoints.

### Intrinsic Subspace Tuning
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
