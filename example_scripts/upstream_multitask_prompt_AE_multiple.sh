cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE

export PYTHONIOENCODING=utf8

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
--recover_multiple_seeds \
--do_predict
