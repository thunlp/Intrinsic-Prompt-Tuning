import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataloader.fewshot_gym_multitask_AE import NLPFewshotGymMultiTaskData_AE

from bart import MyBart
from bartPrompt import MyBartPrompt_AE, MyBartPrompt_ensemble
from utils import freeze_embeds, trim_batch, get_tasks_list

from tqdm import tqdm

def get_params_for_prompt_optimization(module: torch.nn.Module, AE_recover = False, AE_recover_stage_two = False):
    params = []
    params_optimized = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            if AE_recover:
                if not AE_recover_stage_two and "prompt_task" not in t[0]:
                    continue
                elif AE_recover_stage_two and "prompt_embeddings" not in t[0]:
                    continue
            params_optimized.append(t[0])
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
    # if torch.distributed.get_rank() == 0:
    #     print("print params", params)
    return params

def save_ckpt(model, args, global_step, scheduler, optimizer, best_accuracy, AE_recover, logger, prefix=None):
    assert (prefix is not None)
    if AE_recover:
        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items() if 'prompt_task' in k}
    else:
        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items() if 'prompt' in k}
    ckpt_to_save = {
        'global_step': global_step,
        'model': model_state_dict,
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    logger.info('saving checkpoints to: ' + os.path.join(args.output_dir, prefix + "-ckpt.pt"))
    torch.save(ckpt_to_save, os.path.join(args.output_dir, prefix + "-ckpt.pt"))


def run(args, logger, recover_task=None):
    tokenizer = BartTokenizer.from_pretrained(args.model)

    if not args.AE_recover:
        train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
        dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
        test_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    else:
        train_tasks = recover_task
        dev_tasks = recover_task
        test_tasks = recover_task

    logger.info("Training on the following tasks: {}".format(train_tasks))
    logger.info("Dev on the following tasks: {}".format(dev_tasks))

    train_data = NLPFewshotGymMultiTaskData_AE(logger, args, args.train_dir, tasks=train_tasks, data_split="train", data_type="train", is_training=True, is_test=False)
    dev_data = NLPFewshotGymMultiTaskData_AE(logger, args, args.train_dir, tasks=dev_tasks, data_split="dev", data_type="dev", is_training=False, is_test=False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    if args.do_predict:
        logger.info("Test on the following tasks: {}".format(test_tasks))
        test_data = NLPFewshotGymMultiTaskData_AE(logger, args, args.train_dir, tasks=test_tasks, data_split="test", data_type="test", is_training=False, is_test=True)
        test_data.load_dataset(tokenizer)
        test_data.load_dataloader()
    else:
        test_data = None

    if args.do_train:
        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            if args.do_prompt:
                if args.do_AE:
                    model = MyBartPrompt_AE.from_pretrained(args.model, prompt_num=args.prompt_num, intrinsic_dim=args.intrinsic_dim, AE_loss=args.AE_loss, Distil_loss=args.Distil_loss, alpha_AE=args.alpha_AE, AE_type=args.AE_type, AE_recover=args.AE_recover, AE_recover_stage_two=args.AE_recover_stage_two)
                elif args.do_ensemble:
                    model = MyBartPrompt_ensemble.from_pretrained(args.model, prompt_num=args.prompt_num, ontology_idx_1=train_data.idx_1, ontology_idx_2=train_data.idx_2, ontology_general=train_data.task_num, type1_num=args.type1_num, type2_num=args.type2_num, general_num=args.general_num)
                else:
                    print('no model selected!')
                    assert False
                # init prompt weight with random words from vocab
                # performance of random words is worse than random weight 
                # init_ids = list(np.random.randint(0,50265,size=args.prompt_num))
                # model.init_prompt(init_ids)
            else:
                model = MyBart.from_pretrained(args.model)

        # only for debug
        # print(torch.load('/home/qinyujia/CrossFit_prompt/models/few_shot_prompt/singletask-glue-sst2/prompt_weight/glue-sst2_16_100_best.pt'))
        # model.model.encoder.prompt_embeddings.weight.data = torch.load('/home/qinyujia/CrossFit_prompt/models/few_shot_prompt/singletask-glue-sst2/prompt_weight/glue-sst2_16_100_best.pt')

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model, AE_recover=args.AE_recover, AE_recover_stage_two=args.AE_recover_stage_two)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        if args.do_prompt:
            optimizer_grouped_parameters = get_params_for_prompt_optimization(model, AE_recover = args.AE_recover, AE_recover_stage_two = args.AE_recover_stage_two)
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        print("the number of params is ", len(optimizer_grouped_parameters), [p.shape for ps in optimizer_grouped_parameters for p in ps["params"]])
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_rate * args.total_steps,
                                        num_training_steps=args.total_steps)

        best_dev_metric, best_test_metric = train(args, logger, model, train_data, dev_data, test_data, optimizer, scheduler)

    return best_dev_metric, best_test_metric

def train(args, logger, model, train_data, dev_data, test_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    if args.do_AE:
        train_losses_distil = []
        train_losses_AE = []
    best_accuracy = -1.0
    wait_step = 0
    stop_training=False

    if args.AE_recover and not args.AE_recover_random:
        assert args.AE_recover_from_path is not None
        logger.info('recovering from ckpt: ' + args.AE_recover_from_path)
        AE_params = torch.load(args.AE_recover_from_path)
        model_dict = {k: v for (k, v) in model.state_dict().items()}
        # handling the name difference when trained on multiple GPUs / one GPU
        if args.n_gpu > 1 and 'module' not in list(AE_params['model'].keys())[0]:
            AE_params['model'] = {'module.' + k: v for k,v in AE_params['model'].items()}
        elif args.n_gpu == 1 and 'module' in list(AE_params['model'].keys())[0]:
            AE_params['model'] = {k[7: ]: v for k,v in AE_params['model'].items()}
        model_dict.update({k: v.cuda() for (k, v) in AE_params['model'].items()})
        model.load_state_dict(model_dict)

    # load the checkpoint if terminated:
    if os.path.exists(os.path.join(args.output_dir, "last-ckpt.pt")) and os.listdir(args.output_dir):
        logger.info('found existing checkpoints, loading it now...')
        ckpt_to_load = torch.load(os.path.join(args.output_dir, "last-ckpt.pt"))
        if not args.AE_recover_stage_two:
            scheduler.load_state_dict(ckpt_to_load['scheduler'])
            global_step = ckpt_to_load['global_step']
            optimizer.load_state_dict(ckpt_to_load['optimizer'])
            best_accuracy = ckpt_to_load['best_accuracy']
        model_dict = {k: v for (k, v) in model.state_dict().items()}
        model_dict.update({k: v.cuda() for (k, v) in ckpt_to_load['model'].items()})
        model.load_state_dict(model_dict)
        logger.info('step: ' + str(global_step) + ' best_accuracy: ' + str(best_accuracy))
        if global_step > 50000:
            pass
            # wait_step = args.wait_step
        elif best_accuracy > 0:
            wait_step = args.wait_step - 5000
    else:
        logger.info('did not found existing checkpoints: ' + os.path.join(args.output_dir, "last-ckpt.pt"))
    if args.AE_recover_stage_two:
        model.eval()
        prompt_embedding_init = model.generate_prompt_recover(torch.randn(1, 100, 768).cuda(), 1)
        model.model.encoder.prompt_embeddings.weight.data = prompt_embedding_init.squeeze(dim = 0)
        model.train()

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        if global_step > args.total_steps:
            break
        for batch in train_data.dataloader:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            if args.do_AE:
                loss, loss_distil, loss_AE = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True, task_prompt=batch[4])
                train_losses_distil.append(loss_distil.mean().detach().cpu() if args.Distil_loss else 0)
                train_losses_AE.append(loss_AE.mean().detach().cpu() if args.AE_loss else 0)
            elif args.do_ensemble:
                loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True, ontology=[batch[5], batch[6], batch[7]])
            else:
                assert False

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # print('original')
            # print(model.prompt_task.weight)
            # print(model.prompt_task.weight.grad)
            # print('original')
            # print('1')
            # print(torch.sum(torch.abs(model.prompt_embeddings_ontology_type1.weight.grad)))
            # print(torch.mean(torch.abs(model.prompt_embeddings_ontology_type1.weight * (model.prompt_embeddings_ontology_type1.weight.grad != 0))))
            # print(model.prompt_embeddings_ontology_type1.weight.grad)
            # print(torch.sum(model.prompt_embeddings_ontology_type1.weight.grad != 0) / 768)
            # print('2')
            # print(torch.sum(torch.abs(model.prompt_embeddings_ontology_type2.weight.grad)))
            # print(torch.mean(torch.abs(model.prompt_embeddings_ontology_type2.weight * (model.prompt_embeddings_ontology_type2.weight.grad != 0))))
            # print(model.prompt_embeddings_ontology_type2.weight.grad)
            # print(torch.sum(model.prompt_embeddings_ontology_type2.weight.grad != 0) / 768)
            # print('3')
            # print(torch.sum(torch.abs(model.prompt_embeddings_general.weight.grad)))
            # print(torch.mean(torch.abs(model.prompt_embeddings_general.weight * (model.prompt_embeddings_general.weight.grad != 0))))
            # print(model.prompt_embeddings_general.weight.grad)
            # print(torch.sum(model.prompt_embeddings_general.weight.grad != 0) / 768)
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # print('new')
                # print(model.prompt_task.weight.grad)

                # print('new')
                # print('1')
                # print(torch.sum(torch.abs(model.prompt_embeddings_ontology_type1.weight.grad)))
                # print(model.prompt_embeddings_ontology_type1.weight.grad)
                # print(torch.sum(model.prompt_embeddings_ontology_type1.weight.grad != 0) / 768)
                # print('2')
                # print(torch.sum(torch.abs(model.prompt_embeddings_ontology_type2.weight.grad)))
                # print(model.prompt_embeddings_ontology_type2.weight.grad)
                # print(torch.sum(model.prompt_embeddings_ontology_type2.weight.grad != 0) / 768)
                # print('3')
                # print(torch.sum(torch.abs(model.prompt_embeddings_general.weight.grad)))
                # print(model.prompt_embeddings_general.weight.grad)
                # print(torch.sum(model.prompt_embeddings_general.weight.grad != 0) / 768)
                # exit()
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()
            # print('1')
            # print(model.prompt_embeddings_ontology_type1.weight.data)
            # print('2')
            # print(model.prompt_embeddings_ontology_type2.weight.data)
            # print('3')
            # print(model.prompt_embeddings_general.weight.data)

            if (global_step / args.gradient_accumulation_steps) % args.eval_period == 0:
                model.eval()
                logger.info('begin inference!')
                task2score, curr_metric = inference(model if args.n_gpu==1 else model.module, dev_data, args)
                logger.info('dev tasks')
                logger.info(task2score)
                curr_metric_test = 0
                # curr_em = inference(model if args.n_gpu==1 else model.module, dev_data)
                if args.do_AE:
                    logger.info("Step %d, total loss %.2f, distil Loss %.2f, AE Loss %.2f, %s %s, %s %s, current_best: %s, on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        np.mean(train_losses_distil),
                        np.mean(train_losses_AE),
                        "dev all_metric",
                        curr_metric,
                        "test all_metric",
                        curr_metric_test,
                        best_accuracy,
                        epoch))
                    train_losses_distil = []
                    train_losses_AE = []
                elif args.do_ensemble:
                    logger.info("Step %d, total loss %.2f, %s %s, %s %s, current_best: %s, on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        "dev all_metric",
                        curr_metric,
                        "test all_metric",
                        curr_metric_test,
                        best_accuracy,
                        epoch))
                else:
                    assert False
                
                train_losses = []
                if best_accuracy < curr_metric:
                    logger.info("Saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                            ("all_metric", best_accuracy, curr_metric, epoch, global_step))
                    best_accuracy = curr_metric
                    wait_step = 0
                    stop_training = False
                    if not args.AE_recover_stage_two:
                        save_ckpt(model, args, global_step, scheduler, optimizer, best_accuracy, args.AE_recover, logger, prefix='best')
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        if best_accuracy > 0:
                            stop_training = True
                            logger.info('early exiting!!')
                            break
                        elif global_step > 60000:
                            stop_training = True
                            logger.info('early exiting!!')
                            break
                model.train()
                if not args.AE_recover_stage_two:
                    save_ckpt(model, args, global_step, scheduler, optimizer, best_accuracy, args.AE_recover, logger, prefix='last')
        if stop_training:
            break

    if args.do_predict:
        # loading best ckpt for testing
        logger.info('testing on best ckpt now...')
        ckpt_to_load = torch.load(os.path.join(args.output_dir, "best-ckpt.pt"))
        model_dict = {k: v for (k, v) in model.state_dict().items()}
        model_dict.update({k: v.cuda() for (k, v) in ckpt_to_load['model'].items()})
        model.load_state_dict(model_dict)
        model.eval()
        task2score_test, curr_metric_test = inference(model if args.n_gpu==1 else model.module, test_data, args)
        logger.info('test tasks')
        logger.info(task2score_test)
        logger.info(curr_metric_test)

    best_dev_metric = best_accuracy
    best_test_metric = curr_metric_test

    return best_dev_metric, best_test_metric

def inference(model, dev_data, args, save_predictions=False, verbose=False):
    predictions = []
    task_names = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    idx = 0
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch[: -1]] + batch[-1:]
        pad_token_id = dev_data.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        if args.do_AE:
            if args.AE_recover and not args.AE_recover_stage_two:
                task_prompt_recored = model.generate_prompt_recover(batch[2], batch[0].size()[0])
            elif args.AE_recover and args.AE_recover_stage_two:
                task_prompt_recored = None
            else:
                task_prompt_recored = model.generate_prompt(batch[2])
        elif args.do_ensemble:
            task_prompt_recored = model.generate_prompt([batch[3], batch[4], batch[5]])

        # model.model.encoder.prompt_embeddings.weight.data = task_prompt_recored
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 decoder_start_token_id=model.config.bos_token_id,
                                 early_stopping=dev_data.gen_early_stop,
                                 task_prompt_recored=task_prompt_recored,
                                 )
        
        for input_, output, task_prefix in zip(batch[0], outputs, batch[-1]):
            pred = dev_data.decode(output)
            predictions.append(pred)
            task_names.append('_'.join(task_prefix.split('_')[: -2]))
            # print(pred)
            # print(dev_data.data_evaluate[idx])
            # idx += 1
            # input()
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, task_names, verbose=verbose)


