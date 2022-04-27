import os
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from dataloader.fewshot_gym_singletask import NLPFewshotGymSingleTaskData

from bart import MyBart
from bartPrompt import MyBartPrompt
from utils import freeze_embeds, trim_batch

from tqdm import tqdm

def get_params_for_prompt_optimization(module: torch.nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

    # if torch.distributed.get_rank() == 0:
    #     print("print params", params)
    return params

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
        # best_dev_performance, best_model_state_dict = train(args, logger, model, train_data, dev_data, optimizer, scheduler)
        dev_performance = dev(args, logger, model, dev_data, optimizer, scheduler)
    
    return dev_performance

def dev(args, logger, model, dev_data, optimizer, scheduler):
    model.eval()
    curr_performance = inference(model if args.n_gpu==1 else model.module, dev_data)
    logger.info(" %s %s " % (
            dev_data.metric,
            curr_performance,
            ))
    return curr_performance


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
