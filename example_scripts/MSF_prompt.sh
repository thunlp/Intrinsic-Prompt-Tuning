export PYTHONIOENCODING=utf8

total_steps=100000
output_dir=models/multitask_prompt_distil_ae_bs64_dim250_type1_10w_cls
inherit_prompt_path=**path_to_your_trained_prompts**
TASK_SPLIT=dataloader/custom_tasks_splits/train_cls_test_cls.json
train_batch_size=64
intrinsic_dim=250
AE_loss=1
Distil_loss=1
AE_type=1
gradient_accumulation_steps=1

python cli_multitask_AE.py \
--do_train \
--train_dir data \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps $total_steps \
--model bart-base \
--output_dir $output_dir \
--train_batch_size $train_batch_size \
--num_train_epochs 100000 \
--do_prompt \
--do_AE \
--intrinsic_dim $intrinsic_dim \
--eval_period 5000 \
--AE_loss $AE_loss \
--AE_type $AE_type \
--Distil_loss $Distil_loss \
--inherit_prompt_path $inherit_prompt_path \
--gradient_accumulation_steps $gradient_accumulation_steps \
--select_prefix 100 \
--freeze_embeds \
--do_predict
