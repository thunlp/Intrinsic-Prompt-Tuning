***
### Configure Environment

```bash
# Create a new conda environment (optional)
conda create -n ipt python=3.6.9
conda activate ipt
# For building the NLP Few-shot Gym
pip install datasets==1.4.0 py7zr wget
# For reproducing the baseline methods
pip install torch==1.1.0 higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1 rouge==1.0.0
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```
***
### Building the NLP Few-shot Gym

The following code will automatically prepare the data using :hugs: [huggingface datasets](https://github.com/huggingface/datasets), reconstruct the few-shot train/dev sets we sampled, and verify the files with MD5Sum. The processing will take roughly 3 hours. We have already provided the data used in our paper [data_link](https://drive.google.com/file/d/1gooRoE81crfSa5iodzYCkQcGTJVTFEBu/view?usp=sharing).

```bash
cd tasks
# Construct the gym
# --n_proc=10 means the tasks will be prosessed in parallel with 10 subprocesses.
python _build_gym.py --build --n_proc=10
# Verify with MD5Sum
python _build_gym.py --verify
```

If the processing is successful, the verification script will output `[Success] All files are consistent.`

If the processing for any individual task goes wrong (e.g., some datasets are hosted on google drive and there is daily quota issue), you can re-try later by running individual scripts.

```bash
# For example, if you want to construct glue_sst2
cd tasks
python glue_sst2.py
```

Our codes are based on the implementations of this link: https://github.com/INK-USC/CrossFit

(1) For prompt tuning baseline, refer to example_scripts/promptTune_classification_singletasks_0.sh

(2) For multi-task subspace finding, refer to example_scripts/upstream_multitask_prompt_AE.sh

(3) For intrinsic subspace tuning, refer to example_scripts/upstream_multitask_prompt_AE_recover.sh

(4) For combining prompt tuning and IPT, refer to example_scripts/upstream_multitask_prompt_AE_recover_stage2.sh
