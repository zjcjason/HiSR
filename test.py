from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from eval import evaluate
import datasets
from hisr import Berthtc
from structure_encoder import StructureEncoder, get_hierarchy_relations, Tree

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, required=True, help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument("--TP", action="store_true", help='use text propagation or not')
parser.add_argument("--MLA", action="store_true", help='use multi-label attention or not')
parser.add_argument('--extra', default='_macro', choices=['_macro', '_micro'],
                    help='An extra string in the name of checkpoint.')
parser.add_argument("--print", action="store_true", help="print some sample results")
args = parser.parse_args()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location="cpu")
    batch_size = args.batch_size
    device = args.device
    extra = args.extra
    print_sample = args.print
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join('data_new', args.data)
    tp_use = args.TP
    mla_use = args.MLA

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    label_dict = torch.load(
        os.path.join(data_path, 'value_dict.pt'))  # {0: 'CCAT', 1: 'ECAT', 2: 'GCAT', 3: 'MCAT', 4: 'C11'.....}
    label_map = {v: i for i, v in label_dict.items()}  # {'CCAT':0 ......}
    num_class = len(label_dict)

    label_emb = []
    label_attention_mask = []
    for _, i in label_dict.items():
        label_emb.append(i)
    label_embedding = tokenizer(label_emb, padding=True)
    label_emb = label_embedding["input_ids"]
    label_attention_mask = label_embedding["attention_mask"]
    # label_emb = [[element for element in sublist if element not in [102]] for sublist in label_emb]  # 去除102
    label_emb = torch.as_tensor(label_emb, dtype=torch.long)
    label_attention_mask = torch.as_tensor(label_attention_mask, dtype=torch.long)

    dataset = datasets.load_dataset("json",
                                    data_files={"test": 'data_new/{}/{}_test.json'.format(args.data, args.data)})


    def data_map(batch, tokenizer):
        tokenizer_ids = {"input_ids": [], "attention_mask": [], "label": []}
        for t, l in zip(batch["token"], batch["label"]):
            tokens = tokenizer(t, truncation=True, padding="max_length")
            tokenizer_ids["input_ids"].append(tokens["input_ids"])
            tokenizer_ids["attention_mask"].append(tokens["attention_mask"])
            label_list = [0] * num_class
            for idx in l:
                label_list[idx] = 1
            tokenizer_ids["label"].append(label_list)
        return tokenizer_ids


    dataset = dataset.map(lambda x: data_map(x, tokenizer), batched=True, num_proc=16)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test = DataLoader(dataset["test"], batch_size=batch_size)

    graph_model_tp = StructureEncoder(label_map=label_map, data_name=args.data, device=device, graph_model_type="GCN")
    graph_model_mla = StructureEncoder(label_map=label_map, data_name=args.data, device=device, graph_model_type="GCN")
    model = Berthtc.from_pretrained("bert-base-uncased", num_class=num_class, graph_model_tp=graph_model_tp,
                                    graph_model_mla=graph_model_mla, tp_use=tp_use,
                                    mla_use=mla_use, ctr_use=False, ctr_loss_node=None,
                                    ctr_loss_sample=None, margin=None, gold_margin=None, wandb_use=False)

    model.load_state_dict(checkpoint['param'])
    model.to(device)

    model.eval()
    with torch.no_grad(), tqdm(test) as pbar:
        truth = []
        pred = []
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            label_emb = label_emb.to(device)
            label_attention_mask = label_attention_mask.to(device)
            labels = batch["label"].to(device, dtype=torch.long)
            output = model(input_ids, attention_mask, labels, label_emb, label_attention_mask, neg_sample=None)
            pbar.set_description(
                "test_loss:{:.6f}".format(output["loss"].item())
            )
            for l in batch["label"]:
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
    print('test_macro', macro_f1, 'test_micro', micro_f1)
    with open(os.path.join("checkpoints", args.name, "my_htc.log"), "a") as file:
        file.write('test_macro   ')
        file.write(str(macro_f1))
        file.write('     test_micro   ')
        file.write(str(micro_f1))
        file.write('\n')

    # Print out labels and predictions
    # if print_sample:
    #     truth_label = []
    #     pred_label = []
    #     for i in truth:
    #         t = []
    #         for j in i:
    #             t.append(label_dict[j])
    #         truth_label.append(t)
    #     for i in pred:
    #         p = []
    #         for j in range(len(i)):
    #             if i[j] > 0.5:
    #                 p.append(label_dict[j])
    #         pred_label.append(p)
    #
    #     with open(os.path.join("checkpoints", args.name, "print.txt"), "w") as file1:
    #         for i in range(len(truth_label)):
    #             if truth_label[i] != pred_label[i]:
    #                 file1.write("truth:{}".format(truth_label[i]))
    #                 file1.write('\n')
    #                 file1.write("pred:{}".format(pred_label[i]))
    #                 file1.write('\n')
    #                 file1.write('\n')

    # hierarchical_label_dict = get_hierarchy_relations("data/{}/{}.taxonomy".format(args.data,args.data),
    #                                                   label_map,
    #                                                   root=Tree(-1),
    #                                                   fortree=False)
    # pred_id = []
    # for i in pred:
    #     p = []
    #     for j in range(len(i)):
    #         if i[j] > 0.5:
    #             p.append(j)
    #     pred_id.append(p)
    # with open(os.path.join("checkpoints", args.name, "print_id.txt"), "w") as file:
    #     for i in range(len(truth)):
    #         if truth[i] != pred_id[i]:
    #             file.write("truth:{}".format(truth[i]))
    #             file.write('\n')
    #             file.write("pred:{}".format(pred_id[i]))
    #             file.write('\n')
    #             file.write('\n')
