cd /apdcephfs/share_47076/yujiaqin/CrossFit_AE

TASKS=(quarel kilt_ay2 kilt_zsre common_gen ade_corpus_v2-dosage crawl_domain empathetic_dialogues kilt_trex eli5-asks squad-no_context web_questions gigaword race-middle ai2_arc multi_news samsum art lama-conceptnet adversarialqa wiki_bio superglue-copa cos_e definite_pronoun_resolution hellaswag quartz-with_knowledge freebase_qa cosmos_qa piqa lama-squad superglue-multirc break-QDMR spider xsum superglue-record kilt_hotpotqa)

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
