cd /apdcephfs/share_47076/yujiaqin/CrossFit
# TASKS="superglue-rte tweet_eval-sentiment discovery glue-rte superglue-wsc scicite glue-mrpc tweet_eval-stance_hillary tweet_eval-offensive emotion hatexplain glue-cola sick paws ethos-sexual_orientation glue-qqp tweet_eval-emotion sms_spam health_fact glue-mnli imdb ethos-disability glue-wnli scitail trec-finegrained yahoo_answers_topics liar glue-sst2 tweet_eval-stance_abortion circa tweet_eval-stance_climate glue-qnli tweet_eval-emoji ethos-directed_vs_generalized ade_corpus_v2-classification wiki_auto hate_speech_offensive superglue-wic google_wellformed_query tweet_eval-irony ethos-gender onestop_english trec rotten_tomatoes kilt_fever"
IDENTIFIER=few_shot_prompt
GPU=0

export PYTHONIOENCODING=utf8

for TASK in $TASKS
do

echo "Task: $TASK, Few-shot prompt tuning, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=$GPU \
python tune_hps_singletask.py \
--task_dir data/${TASK}/ \
--do_train \
--do_predict \
--learning_rate_list 1e-5 2e-5 5e-5 1e-4 \
--bsz_list 2 4 8 \
--total_steps 100000 \
--eval_period 1000 \
--warmup_steps 10000 \
--model bart-base \
--output_dir models/${IDENTIFIER}/singletask-${TASK} \
--predict_batch_size 32 \
--do_prompt \
--one_prefix \

done
