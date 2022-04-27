cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE

TASKS=(acronym_identification ade_corpus_v2-classification ade_corpus_v2-dosage ade_corpus_v2-effect adversarialqa ag_news ai2_arc anli aqua_rat art aslg_pc12 biomrc blimp-anaphor_gender_agreement blimp-ellipsis_n_bar_1 blimp-sentential_negation_npi_scope boolq break-QDMR break-QDMR-high-level climate_fever codah common_gen commonsense_qa cos_e cosmos_qa crawl_domain crows_pairs dbpedia_14 definite_pronoun_resolution discovery dream e2e_nlg_cleaned eli5-askh eli5-asks eli5-eli5 emo empathetic_dialogues ethos-disability ethos-gender ethos-national_origin ethos-religion ethos-sexual_orientation freebase_qa gigaword glue-cola glue-mnli glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hatexplain hellaswag imdb jeopardy kilt_ay2 kilt_fever kilt_hotpotqa kilt_nq kilt_trex kilt_wow kilt_zsre lama-conceptnet lama-google_re lama-squad lama-trex liar limit math_qa mc_taco medical_questions_pairs multi_news numer_sense openbookqa paws piqa proto_qa qa_srl qasc quail quarel quartz-no_knowledge quartz-with_knowledge quoref race-high race-middle ropes rotten_tomatoes samsum scitail search_qa sms_spam social_i_qa spider squad-no_context superglue-copa superglue-multirc superglue-record superglue-rte superglue-wic superglue-wsc swag tweet_eval-emoji tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary web_questions wiki_bio wiki_qa wiki_split wikisql wino_grande xsum)

TASK=${TASKS[$(( ($TASK_INDEX) % 120 ))]}

export PYTHONIOENCODING=utf8

python cli_multitask_AE.py \
--do_train \
--train_dir data \
--custom_tasks_splits dataloader/custom_tasks_splits_new/individual_tasks/${TASK}.json \
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
