import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

id2alphabet = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}

class WIQA(FewshotGymTextToTextDataset):

    def __init__(self):
        self.hf_identifier = "wiqa"
        self.task_type = "text to text"
        self.license = "unknown"

    def get_choices_and_answer_string(self, datapoint):
        answer_idx = ord(datapoint["answer_label_as_choice"]) - ord("A")

        choices_string = ""
        for i, ans in enumerate(datapoint["choices"]["text"]):
            if i == answer_idx:
                answer_string = ans
            choices_string += " " + id2alphabet[i] + " " + ans
        return choices_string, answer_string

    def map_hf_dataset_to_list(self, hf_dataset, split_name):

        lines = []
        for datapoint in hf_dataset[split_name]:
            choices_string, answer_string = self.get_choices_and_answer_string(datapoint)
            input_text = datapoint["question_stem"].replace("\n", " ") + choices_string + " [SEP] " + " ".join(datapoint["question_para_step"])
            lines.append((input_text, answer_string))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("wiqa")

def main():
    dataset = WIQA()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()