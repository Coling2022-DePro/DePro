
import torch
from torch.utils.data import Dataset

import pandas as pd
import csv


SENTENCE_MAX_LENGTH = 128
MAX_INPUT_LENGTH = SENTENCE_MAX_LENGTH * 2

def check_data(s1, s2, label, args, labels_type):

    if len(s1.split()) == 0 or len(s2.split()) == 0:
        return False
    else:
        return True


def processSentences(tokenizer, samples1_list, samples2_list):
    input_ids = []
    attention_masks = []
    segment_ids = []
    if len(samples1_list) != len(samples2_list):
        raise AssertionError
    
    for idx in range(len(samples1_list)):
        sent1 = samples1_list[idx]
        sent2 = samples2_list[idx]
        encoding = tokenizer.encode_plus(sent1, sent2, max_length=SENTENCE_MAX_LENGTH, truncation=True)
        
        input_id = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        segment_id = encoding['token_type_ids']

        if len(input_id) >= MAX_INPUT_LENGTH:
            input_id = input_id[:MAX_INPUT_LENGTH]
            attention_mask = attention_mask[:MAX_INPUT_LENGTH]
            segment_id = segment_id[:MAX_INPUT_LENGTH]
        else:
            padding_length = MAX_INPUT_LENGTH - len(input_id)

            input_id = input_id + ([tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            segment_id = segment_id + ([0] * padding_length)
        
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        segment_ids.append(segment_id)

    
    return input_ids, attention_masks, segment_ids


class datasets(Dataset):

    def __init__(self, datadir, tokenizer, labels_type, args):
        df_samples = pd.read_csv(datadir, sep='\t', engine='python', quoting=csv.QUOTE_NONE)
        self.samples1_list = []
        self.samples2_list = []
        self.samples_labels = []
        self.label2id = {label: i for i, label in enumerate(labels_type)}

        # similarity,sentence1,sentence2
        for _,row in df_samples.iterrows():
            s1 = str(row['premise'])
            s2 = str(row['hypothesis'])
            label = int(row['label'])
            if check_data(s1, s2, label, args, labels_type):
                self.samples1_list.append(s1)
                self.samples2_list.append(s2)
                self.samples_labels.append(label)
        
        self.len = len(self.samples_labels)
        
        input_ids, attention_masks, segment_ids = processSentences(tokenizer, self.samples1_list, self.samples2_list)
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.segment_ids = segment_ids

    
    def __getitem__(self, index):
        
        label = self.samples_labels[index]

        # return self.input_ids[index], self.attention_masks[index], self.segment_ids[index], self.label2id[label]
        return self.input_ids[index], self.attention_masks[index], self.segment_ids[index], self.samples_labels[index]

    def __len__(self):
        return self.len



class Collate_function:
    def collate(self, batch):
        input_ids, attention_masks, segment_ids, targets = zip(*batch)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return input_ids, attention_masks, segment_ids, targets

    def __call__(self, batch):
        return self.collate(batch)