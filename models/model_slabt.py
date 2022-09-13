import torch
import torch.nn as nn

from transformers.modeling_bert import BertModel


class AutoModelForSlabt(nn.Module):
    def __init__(self, pretrained_path, config, args=None, cls_num=3):
        super(AutoModelForSlabt, self).__init__()

        self.bert_config = config
        self.bert = BertModel.from_pretrained(pretrained_path)
        print(self.bert.config)
        self.fc1 = nn.Linear(self.bert_config.hidden_size, cls_num)


        # for tables
        # for different levels
        self.register_buffer('pre_features', torch.zeros(args.n_feature, args.feature_dim))
        self.register_buffer('pre_weight1', torch.ones(args.n_feature, 1))
        if args.n_levels > 1:
            self.register_buffer('pre_features_2', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_2', torch.ones(args.n_feature, 1))
        if args.n_levels > 2:
            self.register_buffer('pre_features_3', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_3', torch.ones(args.n_feature, 1))
        if args.n_levels > 3:
            self.register_buffer('pre_features_4', torch.zeros(args.n_feature, args.feature_dim))
            self.register_buffer('pre_weight1_4', torch.ones(args.n_feature, 1))
        if args.n_levels > 4:
            print('WARNING: THE NUMBER OF LEVELS CAN NOT BE BIGGER THAN 4')

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[1]
        flatten_features = sequence_output # (batch_size, hidden_size)

        x = self.fc1(flatten_features) # [bsz, classes_num]

        return x, flatten_features
