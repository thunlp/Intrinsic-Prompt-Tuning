cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE
TASKS=(acronym_identification ade_corpus_v2-classification ade_corpus_v2-dosage ade_corpus_v2-effect adversarialqa ag_news ai2_arc anli aqua_rat art aslg_pc12 biomrc blimp-anaphor_gender_agreement blimp-ellipsis_n_bar_1 blimp-sentential_negation_npi_scope boolq break-QDMR break-QDMR-high-level climate_fever codah common_gen commonsense_qa cos_e cosmos_qa crawl_domain crows_pairs dbpedia_14 definite_pronoun_resolution discovery dream e2e_nlg_cleaned eli5-askh eli5-asks eli5-eli5 emo empathetic_dialogues ethos-disability ethos-gender ethos-national_origin ethos-religion ethos-sexual_orientation freebase_qa gigaword glue-cola glue-mnli glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hatexplain hellaswag imdb jeopardy kilt_ay2 kilt_fever kilt_hotpotqa kilt_nq kilt_trex kilt_wow kilt_zsre lama-conceptnet lama-google_re lama-squad lama-trex liar limit math_qa mc_taco medical_questions_pairs multi_news numer_sense openbookqa paws piqa proto_qa qa_srl qasc quail quarel quartz-no_knowledge quartz-with_knowledge quoref race-high race-middle ropes rotten_tomatoes samsum scitail search_qa sms_spam social_i_qa spider squad-no_context superglue-copa superglue-multirc superglue-record superglue-rte superglue-wic superglue-wsc swag tweet_eval-emoji tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary web_questions wiki_bio wiki_qa wiki_split wikisql wino_grande xsum)
IDENTIFIER=few_shot_prompt_base_said_proj
GPU=0

TASK=${TASKS[$(( ($TASK_INDEX) % 120 ))]}

export PYTHONIOENCODING=utf8


echo "Task: $TASK, Few-shot prompt tuning, Identifier: $IDENTIFIER"

CUDA_VISIBLE_DEVICES=$GPU \
python tune_hps_singletask.py \
--task_dir data/${TASK}/ \
--do_train \
--do_predict \
--learning_rate_list 1e-5 \
--bsz_list 4 \
--total_steps 30000 \
--eval_period 1000 \
--warmup_steps 3000 \
--model bart-base \
--output_dir models/${IDENTIFIER}/dim_3/singletask-${TASK} \
--predict_batch_size 32 \
--one_prefix \
--do_said \
--intrinsic_dim 3 \
--wait_step 2000 \
--quiet

