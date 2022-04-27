cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE

TASKS=(ade_corpus_v2-classification ag_news anli climate_fever dbpedia_14 emo ethos-disability ethos-gender ethos-national_origin ethos-religion ethos-sexual_orientation glue-cola glue-mnli glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hatexplain imdb kilt_fever liar medical_questions_pairs paws rotten_tomatoes scitail sms_spam superglue-rte superglue-wic superglue-wsc tweet_eval-emoji tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary wiki_qa)

TASK=${TASKS[$(( ($TASK_INDEX) % 120 ))]}

export PYTHONIOENCODING=utf8

python cli_multitask_AE.py \
--do_train \
--train_dir data \
--custom_tasks_splits dataloader/custom_tasks_splits_new/individual_tasks/${TASK}.json \
--total_steps $total_steps \
--model bart-large_download \
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
