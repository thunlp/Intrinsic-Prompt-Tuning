cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE

TASKS=(acronym_identification ade_corpus_v2-effect aqua_rat aslg_pc12 biomrc blimp-anaphor_gender_agreement blimp-ellipsis_n_bar_1 blimp-sentential_negation_npi_scope boolq break-QDMR-high-level codah commonsense_qa crows_pairs dream e2e_nlg_cleaned eli5-askh eli5-eli5 jeopardy kilt_nq kilt_wow lama-google_re lama-trex limit math_qa mc_taco numer_sense openbookqa proto_qa qa_srl qasc quail quartz-no_knowledge quoref race-high ropes search_qa social_i_qa swag wiki_split wikisql wino_grande)

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
