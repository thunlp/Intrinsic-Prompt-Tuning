cd /apdcephfs/share_47076/yujiaqin/CrossFit

export PYTHONIOENCODING=utf8

TASK_SPLIT=dataloader/custom_tasks_splits/random.json
python cli_maml.py \
--do_train \
--learning_rate 1e-5 \
--output_dir models/upstream-maml \
--custom_tasks_splits ${TASK_SPLIT} \
--total_steps 6000 \
--model bart-base \
--warmup_steps 360 \
--train_batch_size 1 \
--gradient_accumulation_steps 4 \
--num_train_epochs 40

