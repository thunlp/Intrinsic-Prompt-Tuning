cd /apdcephfs/share_47076/yujiaqin/CrossFit

export PYTHONIOENCODING=utf8

TASK_SPLIT=dataloader/custom_tasks_splits/random.json
python cli_multitask.py \
--do_train \
--train_dir data \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 17450 \
--warmup_steps 1047 \
--model bart-base \
--output_dir models/upstream-multitask_prompt \
--train_batch_size 32 \
--num_train_epochs 10 \
--do_prompt
