import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import NluDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, data_raw, labels):
    model.eval()
    test_data = NluDataset(data_raw, annotated=False)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)
    slot_labels = labels['slot_labels']
    intent_labels = labels['intent_labels']
    s_preds = []
    i_preds = []
    epoch_pbar = tqdm(test_dataloader, desc="Prediction", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if b != None and not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, slot_ids, _ = batch
        with torch.no_grad():
            intent_output, slot_output = model(input_ids, segment_ids, input_mask)

        # intent prediction
        intent_output = intent_output.argmax(dim=1)
        intent_output = intent_output.tolist()
        i_preds = i_preds + intent_output
        # slot_evaluate
        slot_output = slot_output.argmax(dim=2)
        slot_output = slot_output.tolist()
        slot_ids = slot_ids.tolist()
        slot_output_string_list = align_predictions(slot_output,slot_ids,slot_labels)
        s_preds = s_preds + slot_output_string_list

        epoch_pbar.update(1)
    epoch_pbar.close()
    sentences = [" ".join(data.words) for data in data_raw]
    # convert id to label
    i_preds = [intent_labels[p] for p in i_preds]
    write_res_to_file(sentences, i_preds, s_preds)
    write_bad_case_to_file(sentences,i_preds,s_preds)


def write_bad_case_to_file(sentences,i_preds,s_preds):
    bad_cases = []
    with open(os.path.join(args.data_dir,'test/intent_seq.out'),'r',encoding='utf-8') as f:
        test_labels = [x.strip() for x in f.readlines()]
    i_labels = [x.split()[0] for x in test_labels]
    s_labels = [" ".join(x.split()[1:]) for x in test_labels]
    for idx, sentence in enumerate(sentences):
        if i_labels[idx] != i_preds[idx]:
            bad_cases.append(sentence+'\n')
            bad_cases.append('Intent prediction: %s . Intent label: %s .\n' % (i_preds[idx], i_labels[idx]))
            bad_cases.append('Slot pred : %s \n' % s_preds[idx])
            bad_cases.append('Slot label: %s \n' % s_labels[idx])
            bad_cases.append('\n')
    with open('bad_case.txt','w',encoding='utf-8') as f:
        f.writelines(bad_cases)

def align_predictions(preds, label_mask, id_to_label):
    res = []
    for pred, mask in zip(preds, label_mask):
        temp = []
        for p, m in zip(pred, mask):
            if m == 1:
                temp.append(id_to_label[p])
        s = ' '.join(temp)
        res.append(s)
    return res


def write_res_to_file(sentences, intent_preds,slot_preds):
    import csv
    with open('preds.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(( 'input_text','intent','slot'))
        for sentence, intent,slot in zip(sentences, intent_preds,slot_preds):
            writer.writerow((sentence,intent,slot))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    test_examples = processor.get_predict_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)

    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.use_crf = args.use_crf
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    model = Model(model_config)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    predict(model, test_data_raw, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/snips/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/snips/+gate/model_best.bin', type=str)
    parser.add_argument("--use_crf", default=False, type=bool)
    parser.add_argument("--max_seq_len", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
