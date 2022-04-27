# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging

import random
import numpy as np
import torch

from utils import get_tasks_list

from run_multitask_AE import run
import json

def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--train_dir", default="data")
    parser.add_argument("--predict_dir", default="data")
    parser.add_argument("--model", default="facebook/bart-base", required=False)
    
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--predict_checkpoint", type=str, default="best-model.pt")

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--do_lowercase", action='store_true', default=False)
    parser.add_argument("--freeze_embeds", action='store_true', default=False)

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")                        
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_rate", default=0.06, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--total_steps", default=100000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10000000000)

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=10000000,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--custom_tasks_splits', type=str, default=None)

    # to prompt tuning
    parser.add_argument("--prompt_num", type=int, default=100)
    parser.add_argument("--do_prompt", action='store_true', help="prompt tuning or not")
    parser.add_argument("--do_inherit_prompt", action='store_true', help="inherit prompt or not")
    parser.add_argument("--inherit_prompt_path", default="models/few_shot_prompt", type=str)
    parser.add_argument("--select_prefix", type=int, default=100, help="-1 means select all seeds of prefix")
    parser.add_argument("--learning_rate_list", nargs="*", type=float, default=[])
    parser.add_argument("--bsz_list", nargs="*", type=int, default=[])

    # to AE tuning
    parser.add_argument("--do_AE", action='store_true', help="auto-encoding or not")
    parser.add_argument("--Distil_loss", type=int, default=1, help="add distillation loss or not")
    parser.add_argument("--AE_loss", type=int, default=1, help="add auto-encoding loss or not")
    parser.add_argument("--AE_type", type=int, default=0, help="specific NN structure for Auto-encoding")
    parser.add_argument("--intrinsic_dim", default=10, type=int,
                        help="intrinsic dimension.")
    parser.add_argument("--alpha_AE", default=200, type=float)
    parser.add_argument("--AE_recover", action='store_true')
    parser.add_argument("--AE_recover_stage_two", action='store_true')
    parser.add_argument("--AE_recover_from_path", default=None, type=str)
    parser.add_argument("--recover_multiple_seeds", action='store_true')
    parser.add_argument("--AE_recover_random", action='store_true')

    # to SAID tuning
    parser.add_argument("--do_said", action='store_true', help="said or not")

    # to intrinsic addapter tuning
    parser.add_argument("--do_adapter", action='store_true', help="said or not")

    # to ensemble tuning
    parser.add_argument("--do_ensemble", action='store_true', help="ensemble or not")
    parser.add_argument("--type1_num", type=int, default=25)
    parser.add_argument("--type2_num", type=int, default=25)
    parser.add_argument("--general_num", type=int, default=50)


    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_dir:
            raise ValueError("If `do_train` is True, then `train_dir` must be specified.")
        if not args.predict_dir:
            raise ValueError("If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.predict_dir:
            raise ValueError("If `do_predict` is True, then `predict_dir` must be specified.")

    logger.info("Using {} gpus".format(args.n_gpu))
    
    if args.AE_recover:
        output_dir = args.output_dir
        task_to_metric_dev_best = {}
        task_to_metric_test_best = {}
        train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
        for task in train_tasks:
            for lr in args.learning_rate_list:
                for bsz in args.bsz_list:
                    logger.info("Recovering ... task={}, lr={}, bsz={} ...".format(task, lr, bsz))
                    args.learning_rate = lr
                    args.train_batch_size = bsz
                    args.output_dir = output_dir + '/' + task + '/' + str(args.select_prefix) + '/' + str(lr) + '_' + str(bsz)
                    '''
                    if os.path.exists(os.path.join(args.output_dir, 'metric.json')):
                        logger.info('said already done, skip')
                        exit()
                    '''
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir, exist_ok=True)
                    best_dev_metric, best_test_metric = run(args, logger, recover_task = [task])
                    param_to_metric_all = {str(lr) + '_' + str(bsz): [best_dev_metric, best_test_metric]}
                    if args.AE_recover_stage_two:
                        json.dump(param_to_metric_all, open(args.output_dir + '/' + 'metric_stage2.json', 'w'))
                    else:
                        json.dump(param_to_metric_all, open(args.output_dir + '/' + 'metric.json', 'w'))
                    if task not in task_to_metric_dev_best:
                        task_to_metric_dev_best[task] = best_dev_metric
                        task_to_metric_test_best[task] = best_test_metric
                    elif task_to_metric_dev_best[task] < best_dev_metric:
                        task_to_metric_dev_best[task] = best_dev_metric
                        task_to_metric_test_best[task] = best_test_metric
                    logger.info('ending training recover task: ' + task)
                    logger.info('dev')
                    logger.info(task_to_metric_dev_best)
                    logger.info('test')
                    logger.info(task_to_metric_test_best)
                    logger.info(np.mean(list(task_to_metric_dev_best.values())))
                    logger.info(np.mean(list(task_to_metric_test_best.values())))
    else:
        run(args, logger)

if __name__=='__main__':
    main()

