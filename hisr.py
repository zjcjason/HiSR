import time

import torch
from torch import nn
from torch.nn import functional
from transformers import BertModel, BertPreTrainedModel
import random
import math
import wandb


class Berthtc(BertPreTrainedModel):
    def __init__(self, config, num_class, graph_model_tp, graph_model_mla, tp_use, mla_use, ctr_use, ctr_loss_node,
                 ctr_loss_sample, margin, gold_margin, wandb_use):
        super(Berthtc, self).__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.num_labels = num_class
        self.classifier = BertClassifier(768, num_class)
        self.loss_fcn = nn.BCEWithLogitsLoss()
        self.pooler = BertPoolingLayer('cls')
        self.tp_use = tp_use
        self.mla_use = mla_use
        self.ctr_use = ctr_use
        self.attention = SelfAttention(768)
        self.TP = TextPropagation(graph_model_tp, self.num_labels)
        self.ML = MultiLable(graph_model_mla, self.num_labels)
        self.ctr_loss_node = ctr_loss_node
        self.ctr_loss_sample = ctr_loss_sample
        self.margin = margin
        self.gold_margin = gold_margin
        self.wandb_use = wandb_use

    def forward(self, input_ids, attention_mask, labels, label_emb, label_attention_mask, neg_sample):
        bert_outputs = self.bert(input_ids, attention_mask, output_hidden_states=True)
        pool_output = bert_outputs[1]  # pool_output (bz,768)  # 0-->last_output   1-->pool_output   2-->hidden_states
        hidden_output = bert_outputs[2][-12:]  # 12 (bz,512,768)
        hidden_output = [self.pooler(layer) for layer in hidden_output]  # 12 (bz,1,768)
        text_feature = torch.cat(hidden_output, dim=1)  # torch.Size([bz, 12, 768])
        text_feature = self.attention(text_feature)  # (bz,1,768)
        # last_output = bert_outputs[0]
        # text_feature = self.pooler(last_output)

        label_bert_outputs = self.bert(label_emb, label_attention_mask, output_hidden_states=True)
        label_emb = label_bert_outputs[2][0]  # hidden_states的第一个，embedding

        target = labels.to(torch.float32)
        loss = 0
        # loss = None
        sign = False

        if not self.tp_use and not self.mla_use:
            logits = self.classifier(pool_output)
            loss += self.loss_fcn(logits.view(-1, self.num_labels), target)
            # loss = self.loss_fcn(logits.view(-1, self.num_labels), target)

        if self.tp_use:
            text_label_f, logits = self.TP(text_feature)
            if self.mla_use:
                sign = True
                logits = self.ML(text_label_f, label_emb, sign)
                loss += self.loss_fcn(logits.view(-1, self.num_labels), target)
            else:
                loss += self.loss_fcn(logits.view(-1, self.num_labels), target)

        if self.mla_use and not self.tp_use:
            logits = self.ML(pool_output, label_emb, sign)
            loss += self.loss_fcn(logits.view(-1, self.num_labels), target)

        if self.wandb_use:
            wandb.log({"loss_classifier": loss})

        if self.ctr_use and self.training:
            logits = torch.sigmoid(logits)
            # 对比学习分两个部分：最难节点和最难序列
            ctr_loss = 0
            loss_sample_all = 0
            loss_node_all = 0
            # 将label的logits置0
            logits_max = logits.masked_fill(target == 1, 0)
            # 选出除label外top节点作为负样本
            _, idxs = torch.topk(logits_max, 10)
            logits_min = logits.masked_fill(target != 1, 1)
            node_gold_score, _ = logits_min.min(dim=1, keepdim=True)
            node_gold_score = node_gold_score.squeeze(-1)  # (batch_size)
            node_neg_score = torch.gather(logits, 1, idxs)  # (batch_size, 10)
            node_loss = RankingLoss(node_neg_score, node_gold_score, self.margin, self.gold_margin, no_cand=True)

            # logits_mean = logits.masked_fill(target != 1, 0)
            # 进入一个sample
            for label, ns, logit in zip(labels, neg_sample, logits):
                """
                g--> 二维列表，label
                ns--> 三维列表,最难序列的负样本
                logit-->tensor.本sample的分数
                idx--> 一维列表，top节点
                e_s-->
                """
                # 选出score最低的label作为正标签
                mask = label == 1  # torch.Size([141])
                gold_score = torch.mean(logit[mask]).unsqueeze(0)
                neg_score = []
                for f1 in ns:
                    score = [sum(logit[i] for i in item) / len(item) for item in f1]
                    all_score = torch.stack(score)
                    # print(all_score)
                    selected, _ = all_score.max(dim=0)
                    neg_score.append(selected)
                    # print(neg_score)
                neg_score = torch.stack(neg_score).unsqueeze(0)
                # print(neg_score)
                # assert exit()
                assert len(ns) == neg_score.shape[1]
                loss_sample = RankingLoss(neg_score, gold_score, self.margin, self.gold_margin) / len(ns)
                # 单个样本最难序列求和并整个batch对比损失求和
                loss_sample_all += loss_sample
                # ctr_loss += (loss_sample + node_loss)
            # 对比损失/bath_size
            # ctr_loss /= len(labels)
            loss_node_all = node_loss
            loss_sample_all /= len(labels)
            # loss += self.ctr_loss * ctr_loss
            loss += (self.ctr_loss_node * loss_node_all + self.ctr_loss_sample * loss_sample_all)
            if self.wandb_use:
                wandb.log({"loss_node": self.ctr_loss_node * loss_node_all,
                           "loss_seq": self.ctr_loss_sample * loss_sample_all,
                           "loss_ctr": self.ctr_loss_node * loss_node_all + self.ctr_loss_sample * loss_sample_all})

        assert not math.isnan(loss)
        return {
            "loss": loss,
            "logits": logits
        }


def RankingLoss(score, summary_score=None, margin=0.001, gold_margin=0, gold_weight=1, no_gold=False,
                no_cand=False):  # (bz. num)
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0)
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


class MultiLable(nn.Module):
    def __init__(self, graph_model_tp, num_labels):
        super(MultiLable, self).__init__()

        self.graph_model = graph_model_tp
        # classifier
        self.linear = nn.Linear(num_labels * 768, num_labels)

        # dropout
        self.dropout = nn.Dropout(p=0.2)

    @staticmethod
    def _soft_attention(text_f, label_f):
        """
        soft attention module
        :param text_f -> torch.FloatTensor, (batch_size, K, dim)
        :param label_f ->  torch.FloatTensor, (N, dim)
        :return: label_align ->  torch.FloatTensor, (batch, N, dim)
        """
        att = torch.matmul(text_f, label_f.transpose(0, 1))  # (batch_size, K, N)
        weight_label = functional.softmax(att.transpose(1, 2), dim=-1)
        label_align = torch.matmul(weight_label, text_f)  # (batch, N, dim)
        return label_align

    def forward(self, text_feature, label_embedding, sign):
        """
        forward pass with multi-label attention
        :param text_feature ->  torch.FloatTensor, (batch_size, K0, text_dim)
        :param label_embedding ->  torch.FloatTensor, ()
        :param sign -> bool  TP+MLA
        :return: logits ->  torch.FloatTensor, (batch, N)
        """
        label_emb = label_embedding[:, 0, :]
        label_emb = label_emb.unsqueeze(0)  # (1,103,768)
        label_feature = self.graph_model(label_emb)  # (1,103,768)
        label_feature = label_feature.squeeze(0)  # (103,768)
        if not sign:
            text_feature = text_feature.unsqueeze(1)

        label_aware_text_feature = self._soft_attention(text_feature,
                                                        label_feature)  # text_feature.shape-->[batch_size,1,dim]   label_feature.shape-->[num_class,dim]

        logits = self.dropout(self.linear(label_aware_text_feature.view(label_aware_text_feature.shape[0], -1)))
        # logits = self.linear(label_aware_text_feature.view(label_aware_text_feature.shape[0], -1))
        return logits


class TextPropagation(nn.Module):
    def __init__(self, graph_model_mla, num_class):
        super(TextPropagation, self).__init__()

        self.num_class = num_class
        self.graph_model = graph_model_mla
        self.transformation = nn.Linear(768, self.num_class * 768)
        # classifier
        self.linear = nn.Linear(self.num_class * 768,
                                self.num_class)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, output):  # output.shape-->(8,768)
        output = output.view(output.shape[0], -1)
        output = self.transformation(output)  # output.shape-->(8,103*768)
        output = output.view(output.shape[0], self.num_class, -1)  # output.shape-->(8,103,768)
        label_wise_text_feature = self.graph_model(
            output)  # output.shape-->torch.Size([batch_size, 103, 768])  label_wise_text_feature.shape-->[batch_size, 103, 768]
        logits = self.dropout(self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1)))
        # logits = self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1))
        return label_wise_text_feature, logits


class BertClassifier(nn.Module):
    def __init__(self, dim, num_class):
        super(BertClassifier, self).__init__()

        self.classifier = nn.Linear(dim, num_class)  # 768-->103
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, text_feature):
        logits = self.dropout(self.classifier(text_feature))
        # logits = self.classifier(text_feature)
        return logits


class BertPoolingLayer(nn.Module):  # 用来处理BERT的输出
    def __init__(self, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':  # 如果 avg 为 'cls'，则选择每个样本的第一个 token（即 [CLS] 标记）的输出。这是一种常见的做法，用于句子级别的分类任务。
            x = x[:, 0, :]
            x = x.unsqueeze(1)
        else:
            x = x.mean(dim=1)  # 如果 avg 不是 'cls'，则对所有 token 的输出进行平均，以获得整个序列的平均表示。
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x 的形状是 (batch_size, seq_length, embed_size)
        Q = self.query(x)  # 查询
        K = self.key(x)  # 键
        V = self.value(x)  # 值

        # 计算注意力分数，形状为 (batch_size, seq_length, seq_length)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention_weights = functional.softmax(attention_scores, dim=-1)

        # 应用注意力权重并求和，形状为 (batch_size, seq_length, embed_size)
        out = torch.matmul(attention_weights, V)

        # 在第二维度上求平均，得到 (batch_size, 1, embed_size)
        out = out.mean(dim=1).unsqueeze(1)

        return out
