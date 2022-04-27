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
--do_ensemble \
--eval_period 5000 \
--inherit_prompt_path $inherit_prompt_path \
--type1_num $type1_num \
--type2_num $type2_num \
--general_num $general_num \
--select_prefix 100
