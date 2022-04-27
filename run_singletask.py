import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataloader.fewshot_gym_singletask import NLPFewshotGymSingleTaskData

from bart import MyBart
from bartPrompt import MyBartPrompt
from utils import freeze_embeds, trim_batch

from intrinsic import intrinsic_dimension, intrinsic_dimension_said

from tqdm import tqdm

def get_params_for_prompt_optimization(module: torch.nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    # if torch.distributed.get_rank() == 0:
    #     print("print params", params)
    return params

def get_params_for_said_optimization(model: torch.nn.Module):
    return [{'params': [p for n, p in model.named_parameters() if p.requires_grad == True], 'weight_decay': 0.0}]

def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.model)

    train_data = NLPFewshotGymSingleTaskData(logger, args, args.train_file, data_type="train", is_training=True)
    dev_data = NLPFewshotGymSingleTaskData(logger, args, args.dev_file, data_type="dev", is_training=False)

    train_data.load_dataset(tokenizer)
    train_data.load_dataloader()

    dev_data.load_dataset(tokenizer)
    dev_data.load_dataloader()

    best_dev_performance = None
    test_performance = None

    best_model_state_dict = None

    if args.do_train:
        if args.checkpoint is not None and args.checkpoint != "None":
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            if args.do_prompt:
                model = MyBartPrompt.from_pretrained(args.model,
                                            state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
            else:
                model = MyBart.from_pretrained(args.model,
                                            state_dict=convert_to_single_gpu(torch.load(args.checkpoint)))
        else:
            if args.do_prompt:
                model = MyBartPrompt.from_pretrained(args.model, prompt_num=args.prompt_num)
                # init prompt weight with random words from vocab
                # performance of random words is worse than random weight 
                # init_ids = list(np.random.randint(0,50265,size=args.prompt_num))
                # model.init_prompt(init_ids)
                # 
                if args.do_inherit_prompt:
                    logger.info("Loading prompt weight from {}".format(args.inherit_prompt_path))
                    init_prompt_weight = torch.load(args.inherit_prompt_path)
                    model.base_model.encoder.prompt_embeddings.weight.data = init_prompt_weight
                    
            else:
                model = MyBart.from_pretrained(args.model)
                if args.do_said:
                    # for par in model.parameters():
                    #     par.requires_grad = False
                    model = intrinsic_dimension(model, args.intrinsic_dim, None, set(), 'fastfood', "cuda")

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']

        if args.do_prompt:
            optimizer_grouped_parameters = get_params_for_prompt_optimization(model)
        elif args.do_said:
            optimizer_grouped_parameters = get_params_for_said_optimization(model)
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        print("the number of params is ", len(optimizer_grouped_parameters), [p.shape for ps in optimizer_grouped_parameters for p in ps["params"]])

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler =  get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps=args.warmup_steps,
                                        num_training_steps=args.total_steps)

        best_dev_performance, best_model_state_dict = train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        if args.do_train and best_model_state_dict is not None:
            if args.do_prompt:
                model = MyBartPrompt.from_pretrained(args.model, state_dict=best_model_state_dict, prompt_num=args.prompt_num)
            elif args.do_said:
                with torch.no_grad():
                    model.intrinsic_parameter.copy_(best_model_state_dict['intrinsic_parameter'])
            else:
                model = MyBart.from_pretrained(args.model,
                                       state_dict=best_model_state_dict)
            logger.info("Loading checkpoint from CPU")
        else:
            checkpoint = os.path.join(args.output_dir, args.predict_checkpoint)
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            if args.do_prompt:
                model = MyBartPrompt.from_pretrained(args.model, state_dict=convert_to_single_gpu(torch.load(checkpoint)), prompt_num=args.prompt_num)
            else:
                model = MyBart.from_pretrained(args.model,
                                       state_dict=convert_to_single_gpu(torch.load(checkpoint)))

                if args.do_said:
                    for par in model.parameters():
                        par.requires_grad = False
                    model = intrinsic_dimension(model, args.intrinsic_dim, None, set(), 'fastfood', "cuda")
            '''
            model = MyBart.from_pretrained(args.model,
                                        state_dict=convert_to_single_gpu(torch.load(checkpoint)))
            '''
            logger.info("Loading checkpoint from {}".format(checkpoint))

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        data_type = "test" if "test" in args.test_file else "dev"
        test_data = NLPFewshotGymSingleTaskData(logger, args, args.test_file, data_type=data_type, is_training=False)

        test_data.load_dataset(tokenizer)
        test_data.load_dataloader()

        test_performance = inference(model, test_data, save_predictions=True, verbose=True)
        logger.info("%s on %s data: %.2f" % (test_data.metric, test_data.data_type, test_performance))

    return best_dev_performance, test_performance, best_model_state_dict

def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_performance = -1.0
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet):
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            
            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # print('original')
            # print(torch.sum(torch.abs(model.model.encoder.prompt_embeddings.weight.grad)))
            # print(torch.mean(torch.abs(model.model.encoder.prompt_embeddings.weight * (model.model.encoder.prompt_embeddings.weight.grad != 0))))
            # print(model.model.encoder.prompt_embeddings.weight.grad)
            # print(torch.sum(model.model.encoder.prompt_embeddings.weight.grad != 0) / 768)

            if global_step % args.gradient_accumulation_steps == 0:
                # print([p for n, p in model.named_parameters() if p.grad is not None and 'prompt' in n])
                # parameters = [p for n, p in model.named_parameters() if p.grad is not None]
                # print(torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters])))
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # print('new')
                # print(torch.sum(torch.abs(model.model.encoder.prompt_embeddings.weight.grad)))
                # print(model.model.encoder.prompt_embeddings.weight.grad)
                # print(torch.sum(model.model.encoder.prompt_embeddings.weight.grad != 0) / 768)
                # exit()
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_performance = inference(model if args.n_gpu==1 else model.module, dev_data)
                logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        dev_data.metric,
                        curr_performance,
                        epoch))
                train_losses = []
                if best_performance < curr_performance:
                    best_model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}

                    # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    # torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))

                    logger.info("Not saving model with best %s: %s -> %s on epoch=%d, global_step=%d" % \
                            (dev_data.metric, best_performance, curr_performance, epoch, global_step))
                    best_performance = curr_performance
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break

                model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break
                
        if stop_training:
            break

    # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    # torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))
    return best_performance, best_model_state_dict

def inference(model, dev_data, save_predictions=False, verbose=False):
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    for i, batch in enumerate(dev_data.dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = dev_data.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 decoder_start_token_id=model.config.bos_token_id,
                                 early_stopping=dev_data.gen_early_stop,)
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    if save_predictions:
        dev_data.save_predictions(predictions)
    return dev_data.evaluate(predictions, verbose=verbose)

