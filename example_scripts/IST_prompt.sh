export PYTHONIOENCODING=utf8

total_steps=100000
learning_rate_list=1e-5
bsz_list=4
AE_recover_from_path=**your_autoencoder_path_trained_during_MSF**
output_dir=models/recover/test
inherit_prompt_path=**path_to_your_trained_prompts**
intrinsic_dim=100
AE_loss=0
Distil_loss=1
AE_type=1
gradient_accumulation_steps=1
select_prefix=21

TASK_SPLIT=dataloader/custom_tasks_splits/random.json

CUDA_VISIBLE_DEVICES=1 python cli_multitask_AE.py \
--do_train \
--train_dir data \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps $total_steps \
--model bart-base \
--output_dir $output_dir \
--learning_rate_list $learning_rate_list \
--bsz_list $bsz_list \
--num_train_epochs 100000 \
--do_prompt \
--do_AE \
--intrinsic_dim $intrinsic_dim \
--eval_period 1000 \
--AE_loss $AE_loss \
--AE_type $AE_type \
--Distil_loss $Distil_loss \
--inherit_prompt_path $inherit_prompt_path \
--AE_recover \
--AE_recover_from_path $AE_recover_from_path \
--freeze_embeds \
--gradient_accumulation_steps $gradient_accumulation_steps \
--select_prefix $select_prefix \
--do_predict \
--wait_step 100000
