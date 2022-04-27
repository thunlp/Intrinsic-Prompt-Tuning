import os
import json
import re
import string
import random

import numpy as np

from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from .utils import MyQADataset, MyQAPromptDataset, MyQAPromptDataset_AE, MyDataLoader
from .metrics import METRICS, evaluate

class NLPFewshotGymMultiTaskData_AE(object):

    def __init__(self, logger, args, data_path, tasks, data_split, data_type, is_training, is_test=False):
        self.data_path = data_path
        self.data_type = data_type
        
        self.data = []

        # keep "sorted" so that things are consistent across machines
        c1 = 0
        c2 = 0
        self.ontology = json.load(open('ontology.json', 'r'))
        self.ontology = {k: v[0].split('/') for k, v in self.ontology.items()}
        self.task2id = {}
        self.task_num = 0
        for k, v in self.ontology.items():
            self.task2id[k] = self.task_num
            self.task_num += 1
        ontology_num = np.max([len(v) for v in list(self.ontology.values())])
        # we use two ontologies now
        assert ontology_num == 2
        self.ontology_type1 = {}
        self.idx_1 = 0
        self.ontology_type2 = {}
        self.idx_2 = 0
        for v in list(self.ontology.values()):
            if v[0] not in self.ontology_type1:
                self.ontology_type1[v[0]] = self.idx_1
                self.idx_1 += 1
            if v[1] not in self.ontology_type2:
                self.ontology_type2[v[1]] = self.idx_2
                self.idx_2 += 1
        for task in sorted(tasks):
            task_dir = os.path.join(self.data_path, task)
            logger.info(task_dir)
            files = sorted(os.listdir(task_dir))

            prefixes = []
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                if args.select_prefix >= 0:
                    if '_' + str(args.select_prefix) not in prefix:
                        continue
                
                with open(os.path.join(task_dir, prefix + "_train.tsv"), encoding='utf-8') as fin:
                    lines = fin.readlines()

                train_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    train_examples.append((d[0], d[1:]))

                with open(os.path.join(task_dir, prefix + "_dev.tsv"), encoding='utf-8') as fin:
                    lines = fin.readlines()
                    
                dev_examples = []
                for line in lines:
                    d = line.strip().split("\t")
                    dev_examples.append((d[0], d[1:]))

                if is_test:
                    with open(os.path.join(task_dir, prefix + "_test.tsv"), encoding='utf-8') as fin:
                        lines = fin.readlines()
                        
                    test_examples = []
                    for line in lines:
                        d = line.strip().split("\t")
                        test_examples.append((d[0], d[1:]))
                else:
                    test_examples = []

                # add auto-encoding prompts
                task_prompt = []
                prompt_weight_dir = os.path.join(args.inherit_prompt_path, 'singletask-' + task, 'prompt_weight')
                load_flag = False
                for prompt_dir in os.listdir(prompt_weight_dir):
                    '''
                    if '_' + str(args.select_prefix) + '_' not in prompt_dir:
                        continue
                    '''
                    if not args.recover_multiple_seeds and 'best' not in prompt_dir:
                        continue
                    if args.recover_multiple_seeds and 'best' in prompt_dir:
                        continue
                    task_prompt.append(torch.load(os.path.join(prompt_weight_dir, prompt_dir)))
                    load_flag = True
                if load_flag:
                    c1 += 1
                else:
                    logger.info('did not load the ckpt:' + prompt_weight_dir + ' check later')
                    task_prompt.append(torch.randn(100, 768).float() / 150)
                    c2 += 1
                self.data.append({
                    "task_name": task,
                    "task_prefix": prefix,
                    "train_examples": train_examples,
                    "dev_examples": dev_examples,
                    "test_examples": test_examples,
                    "task_prompt": task_prompt,
                    "ontology": [self.ontology_type1[self.ontology[task][0]], self.ontology_type2[self.ontology[task][1]]],
                })
        logger.info('found prompt in ' + str(c1) + ' tasks')
        logger.info('did not found prmopt in ' + str(c2) + ' tasks')
        self.data_split = data_split
        self.is_training = is_training
        self.logger = logger
        self.args = args

        self.metric = METRICS
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.load = not args.debug

        self.gen_early_stop = False

        self.do_AE = self.args.do_AE

        self.do_ensemble = self.args.do_ensemble
        self.type1_num = args.type1_num
        self.type2_num = args.type2_num
        self.general_num = args.general_num

        self.data_evaluate = []

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type + "-multi-{}-{}.json".format(split_identifier, postfix)
        )
        
        if self.load and os.path.exists(preprocessed_path) and (not self.do_AE) and (not self.do_ensemble):
            # load preprocessed input
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r", encoding='utf-8') as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata = json.load(f)
        else:
            self.logger.info("Start tokenizing ... {} instances".format(len(self.data)))

            inputs = []
            outputs = []
            task_prefix = []
            task_names = []
            all_task_prompts = {}
            ontology = []

            idx = 0
            for task in self.data:
                idx += 1
                if self.args.debug:
                    if idx >= 10:
                        break
                task_name = task["task_name"]
                all_task_prompts[task["task_prefix"]] = task["task_prompt"]

                if self.data_split == "train" or self.data_split == "all":
                    for dp in task["train_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        outputs.append([" " + item for item in dp[1]])
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        ontology.append(task["ontology"])
                        self.data_evaluate.append(dp)
                if self.data_split == "dev" or self.data_split == "all":
                    for dp in task["dev_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        outputs.append([" " + item for item in dp[1]])
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        ontology.append(task["ontology"])
                        self.data_evaluate.append(dp)
                if self.data_split == "test":
                    for dp in task["test_examples"]:
                        inputs.append(" [{}] {}".format(task_name, dp[0]))
                        outputs.append([" " + item for item in dp[1]])
                        task_prefix.append(task["task_prefix"])
                        task_names.append(task_name)
                        ontology.append(task["ontology"])
                        self.data_evaluate.append(dp)

            outputs, metadata = self.flatten(outputs) # what is metadata?

            self.logger.info("Printing 3 examples")
            for i in range(3):
                self.logger.info(task_names[i])
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])
                self.logger.info(task_prefix[i])
                self.logger.info(ontology[i])

            if self.args.do_lowercase:
                inputs = [input0.lower() for input0 in inputs]
                outputs = [output0.lower() for output0 in outputs]
            if self.args.append_another_bos:
                inputs = ["<s> "+input0 for input0 in inputs]
                outputs = ["<s> " +output0 for output0 in outputs]

            self.logger.info("Tokenizing Input ...")
            tokenized_input = tokenizer.batch_encode_plus(inputs,
                                                         pad_to_max_length=True,
                                                         max_length=self.args.max_input_length)
            self.logger.info("Tokenizing Output ...")
            tokenized_output = tokenizer.batch_encode_plus(outputs,
                                                       pad_to_max_length=True,
                                                       max_length=self.args.max_output_length)

            input_ids, attention_mask = tokenized_input["input_ids"], tokenized_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = tokenized_output["input_ids"], tokenized_output["attention_mask"]

            if self.load and (not self.do_AE) and (not self.do_ensemble):
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                with open(preprocessed_path, "w", encoding='utf-8') as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)

        if self.args.do_prompt and not (self.args.do_AE or self.args.do_ensemble):
            self.dataset = MyQAPromptDataset(input_ids, attention_mask,
                                            decoder_input_ids, decoder_attention_mask,
                                            in_metadata=None, out_metadata=metadata,
                                            is_training=self.is_training, prompt_num=self.args.prompt_num)
        elif self.args.do_prompt and (self.args.do_AE or self.args.do_ensemble):
            self.dataset = MyQAPromptDataset_AE(input_ids, attention_mask,
                                            decoder_input_ids, decoder_attention_mask,
                                            task_prefix, task_names, all_task_prompts, ontology,
                                            in_metadata=None, out_metadata=metadata,
                                            is_training=self.is_training, prompt_num=self.args.prompt_num,
                                            type1_num=self.type1_num,
                                            type2_num=self.type2_num,
                                            general_num=self.general_num,
                                            task2id=self.task2id)
        elif self.args.do_adapter:
            self.dataset = MyQADataset(input_ids, attention_mask,
                                            decoder_input_ids, decoder_attention_mask,
                                            in_metadata=None, out_metadata=metadata,
                                            is_training=self.is_training,
                                            task_prefix=task_prefix,
                                            do_adapter=self.args.do_adapter)
        else:
            # never enter this line.
            assert False
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions, task_names, verbose=False):
        assert len(predictions)==len(self.data_evaluate), (len(predictions), len(self.data_evaluate))
        predictions = [prediction.strip() for prediction in predictions]
        task2id = {}
        for idx, task_name in enumerate(task_names):
            if task_name not in task2id:
                task2id[task_name] = []
            task2id[task_name].append(idx)
        task2score = {}
        for task, ids in task2id.items():
            task2score[task] = evaluate([predictions[x] for x in ids], [self.data_evaluate[x] for x in ids], self.metric[task])
        return task2score, np.mean(list(task2score.values()))

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))

        predictions = ['n/a' if len(prediction.strip())==0 else prediction for prediction in predictions]
        prediction_text = [prediction.strip()+'\n' for prediction in predictions]
        save_path = os.path.join(self.args.output_dir, "{}_predictions.txt".format(self.args.prefix))
        with open(save_path, "w", encoding='utf-8') as f:
            f.writelines(prediction_text)
        
        self.logger.info("Saved prediction in {}".format(save_path))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([f1_score(prediction, gt) for gt in groundtruth])
    return f1_score(prediction, groundtruth)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))



