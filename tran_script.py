from fastchat.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()
import json
from torch.utils.data import Dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AdamW
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
import wandb
import os
# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_START_METHOD'] = 'thread'
os.environ["WANDB_API_KEY"] = ''
os.environ['WANDB_DEBUG'] = 'all'

IGNORE_TOKEN_ID = -100

class MixData(Dataset):
    def __init__(self, dataset, ratio, tokenizer):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.data_size = [len(c) for c in self.dataset]
        ratio = [r if isinstance(r, int) else s for r, s in zip(ratio, self.data_size)]
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.sample_size = [int(self.data_size[0] / self.ratio[0] * r) for r in self.ratio]
        print(self.data_size, self.sample_size, [c1 / c2 for c1, c2 in zip(self.sample_size, self.data_size)])

    @staticmethod
    def rounder(number):
        rand = np.random.rand()
        if rand < number - int(number):
            return int(number) + 1
        else:
            return int(number)

    @staticmethod
    def choice_index(number, sample_size):
        for i in range(len(sample_size)):
            if number < sum(sample_size[:i + 1]):
                return i, number - sum(sample_size[:i])

    def __getitem__(self, index):
        corpus_id, index = self.choice_index(index, self.sample_size)
        rand = np.random.rand()
        index = self.rounder((index + rand) / self.sample_size[corpus_id] * self.data_size[corpus_id])
        index = min(index, len(self.dataset[corpus_id]) - 1)
        return self.dataset[corpus_id][index]

    def __len__(self):
        return sum(self.sample_size)

    def set_ratio(self, ratio):
        self.ratio = ratio
        self.data_size = [len(c) for c in self.dataset]
        self.sample_size = [int(self.data_size[0] / self.ratio[0] * r) for r in self.ratio]
        print(self.data_size, self.sample_size, [c1 / c2 for c1, c2 in zip(self.sample_size, self.data_size)])

    def collate_fn(self, data):
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = {
            'input_ids': input_ids.long(),
            'labels': labels.long(),
            'attention_mask': attention_mask.long(),
        }
        return features

def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)

def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]

dummy_message = [{"role": "user", "content": "Who are you?"},
                 {"role": "assistant", "content": "I am chatgpt"},
                 {"role": "user", "content": "What can you do?"},
                 {"role": "assistant", "content": "I can chat with you."}]

def tokenize(messages, tokenizer):
    input_ids = []
    labels = []
    system = "You are a helpful assistant."
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    input_ids += system_ids
    labels += [IGNORE_TOKEN_ID] * len(system_ids)
    for i, turn in enumerate(messages):
        role = turn['role']
        content = turn['content']
        content = content.strip()
        role_ids = tokenizer.encode(role+"\n", add_special_tokens=False)
        content_ids = tokenizer.encode(content+"\n", add_special_tokens=False, truncation=True,
                                       max_length=tokenizer.model_max_length)
        input_ids += role_ids + content_ids
        labels += [IGNORE_TOKEN_ID] * len(role_ids) + content_ids
    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]
    trunc_id = last_index(labels, IGNORE_TOKEN_ID) + 1
    input_ids = input_ids[:trunc_id]
    labels = labels[:trunc_id]
    if len(labels) == 0:
        return tokenize(dummy_message, tokenizer)
    input_ids = safe_ids(input_ids, tokenizer.vocab_size, tokenizer.pad_token_id)
    labels = safe_ids(labels, tokenizer.vocab_size, IGNORE_TOKEN_ID)
    return input_ids, labels
class VicunaData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        input_ids, labels = tokenize(item, self.tokenizer)
        return torch.tensor(input_ids), torch.tensor(labels)

    def collate_fn(self, data):
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        features = {
            'input_ids': input_ids.long(),
            'labels': labels.long(),
            'attention_mask': attention_mask.long(),
        }
        return features

def main():
    accelerator = Accelerator(gradient_accumulation_steps=4,log_with="wandb")
    batch_size = 2
    save_path = ""
    model_name = ""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="right", model_max_length=4096,trust_remote_code=True,pad_token='<|endoftext|>')
    tokenizer.pad_token = '<|endoftext|>'
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    data = []
    with open("training_data.json", 'r') as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    dataset = VicunaData(data, tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=batch_size, num_workers=0, shuffle=True,drop_last=True)
    optimizer = AdamW(model.parameters(), 1e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    wandb.login()
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="wandb-test", 
        config={
            "epochs": 5,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "model_max_length":4096
        },
        init_kwargs={"wandb": {"entity": "endNone"}}
        )
    for epoch in range(5):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch_idx, batch in enumerate(tk0):
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                loss_report.append(accelerator.gather(loss).mean().item())
                tk0.set_postfix(train_loss=np.mean(loss_report[-100:]))  
                accelerator.log({"train_loss":  loss.item()}, step=batch_idx)
        accelerator.wait_for_everyone()
        model.save_checkpoint(f'{save_path}/{epoch}')
    accelerator.end_training()
if __name__ == '__main__':
    main()