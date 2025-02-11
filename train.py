import argparse
import os
import time

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from transformers import AutoTokenizer
from eval import evaluate
from structure_encoder import StructureEncoder
from hisr import Berthtc
from utils import Saver, slot2depth, setup_logging, seed_everything, WarmupCosineLR
import wandb

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='A name for different runs.')
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--arch", type=str, default="bert-base-uncased")
    parser.add_argument("--data", type=str, default="wos")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--early_stop", type=int, default=6, help='Epoch before early stop.')
    parser.add_argument("--TP", action="store_true", help='use text propagation or not')
    parser.add_argument("--MLA", action="store_true", help='use multi-label attention or not')
    parser.add_argument("--wandb", action='store_true', help="use wandb monitor model")
    parser.add_argument('--warmup', default=2, type=int, help='Warmup steps.')
    parser.add_argument("--contrast", action="store_true", help='use contrast_loss or not')
    parser.add_argument('--seed', default=1020, type=int)
    parser.add_argument("--ctr_l_n", type=float, default=0.01, help='contrast_loss weight in node')
    parser.add_argument("--ctr_l_s", type=float, default=0.01, help='contrast_loss weight in sample')
    parser.add_argument("--step", type=int, default=1, help='accumulation_steps ')
    parser.add_argument("--margin", type=float, default=0.15, help="ctr_loss margin")
    parser.add_argument("--gold_margin", type=float, default=0.5, help="ctr_loss gold_margin")
    parser.add_argument("--project", type=str, default="berthtc", help='Project name')
    return parser


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    # seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join("data_new", args.data)
    batch_size = args.batch_size
    device = args.device
    tp_use = args.TP
    mla_use = args.MLA
    ctr_use = args.contrast
    ctr_loss_node = args.ctr_l_n
    ctr_loss_sample = args.ctr_l_s
    margin = args.margin
    gold_margin = args.gold_margin
    accumulation_steps = args.step
    wandb_use = args.wandb
    if wandb_use:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project + args.data,

            # track hyperparameters and run metadata
            config=args
        )
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    logger = setup_logging(os.path.join('checkpoints', args.name, 'my_htc.log'))
    # logger = logging.getLogger('my_htc')
    logger.info(args)

    label_dict = torch.load(
        os.path.join(data_path, "value_dict.pt"))  # {0: 'CCAT', 1: 'ECAT', 2: 'GCAT', 3: 'MCAT', 4: 'C11'.....}
    num_class = len(label_dict)  # number for classes
    label_map = {v: i for i, v in label_dict.items()}  # use for structure_encoder  {'CCAT':0, "ECAT":1,....}
    label_emb = []  # label word vector
    label_attention_mask = []
    for _, i in label_dict.items():
        label_emb.append(i)
    label_embedding = tokenizer(label_emb, padding=True)
    label_emb = label_embedding["input_ids"]
    label_attention_mask = label_embedding["attention_mask"]
    # label_emb = [[element for element in sublist if element not in [102]] for sublist in label_emb]  # remoce 102
    label_emb = torch.as_tensor(label_emb, dtype=torch.long)  # to tensor
    label_attention_mask = torch.as_tensor(label_attention_mask, dtype=torch.long)

    depth = slot2depth(os.path.join(data_path, "slot.pt"))

    dataset = datasets.load_dataset("json",
                                    data_files={
                                        "train": 'data_new/{}/{}_train_ctr_com_0603.json'.format(args.data, args.data),
                                        'dev': 'data_new/{}/{}_dev_ctr_com_0603.json'.format(args.data, args.data)})


    # dataset = datasets.load_dataset("json",
    #                                 data_files={
    #                                     "train": 'data_new/{}/{}_dev_ctr_com_2.json'.format(args.data, args.data),
    #                                     'dev': 'data_new/{}/{}_train_ctr_com_2.json'.format(args.data, args.data)})

    # dataset_train = datasets.load_dataset("json", 'data/{}/{}_train.json'.format(args.data, args.data))
    # dataset_dev = datasets.load_dataset("json", 'data/{}/{}_dev.json'.format(args.data, args.data))

    def data_map(batch, tokenizer):
        tokenizer_ids = {"input_ids": [], "attention_mask": [], "label": [], "neg_sample": []}
        for t, l, n in zip(batch["token"], batch["label"], batch["neg_sample"]):
            tokens = tokenizer(t, truncation=True, padding="max_length")
            tokenizer_ids["input_ids"].append(tokens["input_ids"])
            tokenizer_ids["attention_mask"].append(tokens["attention_mask"])
            label_list = [0] * num_class
            for idx in l:
                label_list[idx] = 1
            tokenizer_ids["label"].append(label_list)

            # gp = [[0] * num_class for _ in range(4)]
            # for row_idx, cols in enumerate(g):
            #     for col_idx in cols:
            #         gp[row_idx][col_idx] = 1
            tokenizer_ids["neg_sample"].append(n)
        return tokenizer_ids


    dataset = dataset.map(lambda x: data_map(x, tokenizer), batched=True, num_proc=4)
    logger.info("dataset.map finished!!")
    # print(dataset["train"][:1])
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label", "neg_sample"])


    # dataset.set_format(columns=["input_ids", "attention_mask", "label", "group", "neg_sample"])
    # print(dataset["train"][:10]["group"])
    # print(dataset["train"][:1]["extra_sample"])
    # assert exit()

    def mycollate(batch):
        return batch


    def tensor2list(tensor):
        converted_list = []
        for item in tensor:
            if isinstance(item, list):
                # 如果是列表，递归转换
                converted_list.append(tensor2list(item))
            elif isinstance(item, torch.Tensor):
                # 如果是张量，转换为列表
                converted_list.append(item.tolist())
            else:
                # 否则，保持原样
                converted_list.append(item)
        return converted_list


    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=mycollate)
    dev = DataLoader(dataset['dev'], batch_size=32, shuffle=False, collate_fn=mycollate)
    logger.info("data_load finished!!")

    graph_model_tp = StructureEncoder(label_map=label_map, data_name=args.data, device=device, graph_model_type="GCN")
    graph_model_mla = StructureEncoder(label_map=label_map, data_name=args.data, device=device, graph_model_type="GCN")
    model = Berthtc.from_pretrained("bert-base-uncased", num_class=num_class, graph_model_tp=graph_model_tp,
                                    graph_model_mla=graph_model_mla, tp_use=tp_use,
                                    mla_use=mla_use, ctr_use=ctr_use, ctr_loss_node=ctr_loss_node,
                                    ctr_loss_sample=ctr_loss_sample, margin=margin, gold_margin=gold_margin,
                                    wandb_use=wandb_use)

    logger.info("Berthtc finished!!")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    # cosine_lr = WarmupCosineLR(optimizer, 1e-6, 5e-4, args.warmup, 15, 0.1)
    cosine_lr = CosineAnnealingWarmRestarts(optimizer, int(0.5*len(train)), eta_min=1e-6, T_mult=2)
    save = Saver(model, optimizer, args)

    best_score_macro = 0
    best_score_micro = 0
    early_stop_count = 0
    loop = 1

    if not tp_use and not mla_use:
        logger.info("BERT only!!!!")
    if tp_use and not mla_use:
        logger.info("BERT + TP!!!!")
    if not tp_use and mla_use:
        logger.info("BERT + MLA!!!!")
    if tp_use and mla_use:
        logger.info("BERT + TP + MLA！！！")
    if ctr_use:
        logger.info("add contrast_loss")

    logger.info("start training!!")
    start = time.time()

    for epoch in range(1000):
        if early_stop_count >= args.early_stop:
            logger.info("Early stop!")
            logger.info(f"best_score_macro:{best_score_macro}     best_score_micro:{best_score_micro}")
            end = time.time()
            logger.info("run time is {:.6f}".format(end - start))
            break
        model.train()
        with tqdm(train) as p_bar:
            step = 0
            for batch in p_bar:
                # print(batch)
                # input_ids = batch["input_ids"].to(device, dtype=torch.long)
                input_ids = torch.stack([item["input_ids"] for item in batch]).to(device, dtype=torch.long)
                # attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
                attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device, dtype=torch.long)
                label_emb = label_emb.to(device)
                label_attention_mask = label_attention_mask.to(device)
                # labels = batch["label"].to(device, dtype=torch.long)
                labels = torch.stack([item["label"] for item in batch]).to(device, dtype=torch.long)
                # label_idx = batch["label_idx"].to(device, dtype=torch.long)
                # group = [tensor2list(item["group"]) for item in batch]
                neg_sample = [tensor2list(item["neg_sample"]) for item in batch]
                output = model(input_ids, attention_mask, labels, label_emb, label_attention_mask, neg_sample)
                if wandb_use:
                    wandb.log({"loss": output["loss"]})
                output["loss"].backward()
                p_bar.set_description(
                    "loop_{} train_loss:{:.6f}".format(loop, output["loss"].item())
                )
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(p_bar):
                    optimizer.step()
                    optimizer.zero_grad()
                step += 1
                cosine_lr.step()
        p_bar.close()

        model.eval()
        with torch.no_grad(), tqdm(dev) as pbar:
            truth = []
            pred = []
            for batch in pbar:
                # input_ids = batch["input_ids"].to(device, dtype=torch.long)
                input_ids = torch.stack([item["input_ids"] for item in batch]).to(device, dtype=torch.long)
                # attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
                attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device, dtype=torch.long)
                # labels = batch["label"].to(device, dtype=torch.long)
                labels = torch.stack([item["label"] for item in batch]).to(device, dtype=torch.long)
                label_attention_mask = label_attention_mask.to(device)
                # label_idx = batch["label_idx"].to(device, dtype=torch.long)
                output = model(input_ids, attention_mask, labels, label_emb, label_attention_mask, neg_sample=None)
                pbar.set_description(
                    "eval_loss:{:.6f}".format(output["loss"].item())
                )
                # for l in batch["label"]:
                for l in labels:
                    t = []
                    for i in range(l.size(0)):
                        if l[i].item() == 1:
                            t.append(i)
                    truth.append(t)
                for l in output["logits"]:
                    pred.append(torch.sigmoid(l).tolist())
        pbar.close()
        scores = evaluate(pred, truth, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        logger.info(f'macro:{macro_f1}   micro:{micro_f1}    early_stop:{early_stop_count}')

        early_stop_count += 1
        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            logger.info(f"best_score_macro:{best_score_macro}")
            if wandb_use:
                wandb.log({"best_score_macro": best_score_macro})
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            logger.info(f"best_score_micro:{best_score_micro}")
            if wandb_use:
                wandb.log({"best_score_micro": best_score_micro})
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            early_stop_count = 0
        loop += 1
