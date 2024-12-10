import argparse
import random
import os
import glob
import numpy as np
import pandas as pd
from functools import partial

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForTextToWaveform, EncodecModel, BitsAndBytesConfig, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return bnb_config

def read_audio(file_path):
    try:
        audio, sr = torchaudio.load(file_path, format="wav")
        if sr != 32000:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=32000)
            sr = 32000
    except Exception as e:
        print(f"Error reading {file_path}\nError: {e}")
        return None
    return audio.squeeze().numpy()

def get_text_audio(pref_audio_dir, rej_audio_dir, text_desfile):
    pref_list = []
    rej_list = []
    text_list = []
    description = pd.read_csv(text_desfile, header=0)
    for _, row in description.iterrows():
        pref_file = os.path.join(pref_audio_dir, f"{row['id']:04d}_*.wav")
        rej_file = os.path.join(rej_audio_dir, f"{row['prompt']}_*.wav")
        pref_audio = [read_audio(files) for files in glob.glob(pref_file)[1:6]]
        rej_audio = [read_audio(files) for files in glob.glob(rej_file)[1:6]]
        if len(pref_audio) == len(rej_audio):
            text = description[description['id']==row['id']]['prompt'].values[0]
            text_list.extend([text] * len(pref_audio))
            pref_list.extend(pref_audio)
            rej_list.extend(rej_audio)
    text_list, pref_list, rej_list = zip(*[(text, pref, rej) for text, pref, rej in zip(text_list, pref_list, rej_list) if (pref.shape[0] == 800_000 and rej.shape[0] == 817_280)])
    pair_dict = {'text': list(text_list), 'pref_audio': list(pref_list), 'rej_audio': list(rej_list)}
    return pair_dict

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def DPO_loss(model_prefered_logprob, model_disprefered_logprob, ref_prefered_logprob, ref_disprefered_logprob, beta=0.5):
    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    labels = labels.view(-1, labels.shape[-1])
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

class DPOTrainer():
    def __init__(self, args, dataset):
        self.args = args
        self.encoder = EncodecModel.from_pretrained(self.args.encoder_path)
        self.processor = AutoProcessor.from_pretrained(self.args.processor_path)
        self.bnb_config = get_bnb_config()
        self.model = AutoModelForTextToWaveform.from_pretrained(self.args.model_path, load_in_8bit=True) # quantization_config=self.bnb_config)
        self.ref_model = AutoModelForTextToWaveform.from_pretrained(self.args.model_path, load_in_8bit=True) # quantization_config=self.bnb_config)
        self.model.freeze_text_encoder()
        self.model.freeze_audio_encoder()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset

    def load_model(self):
        config = LoraConfig(
                            r=self.args.lora_r, 
                            lora_alpha=self.args.lora_alpha, # default=32 
                            # target_modules=["v_proj",
                            #                 "q_proj",],
                            target_modules=find_all_linear_names(self.args, self.model),
                            lora_dropout=self.args.lora_dropout,  # default=0.1
                            bias="none", 
                        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = bnb.optim.PagedAdamW(optimizer_grouped_parameters, lr=self.args.lr)

    def feature_preparing(self, batch):
        # convert waveform to audio tokens
        print(torch.tensor(batch["pref_audio"]).shape)
        print(torch.tensor(batch["rej_audio"]).shape)
        pref_audio = self.processor(
            audio=batch["pref_audio"],
            sampling_rate=32000,
            padding=True,
            return_tensors="pt",
        ).input_values

        rej_audio = self.processor(
            audio=batch["rej_audio"],
            sampling_rate=32000,
            padding=True,
            return_tensors="pt",
        ).input_values

        return {
            "input_text": batch["text"],
            "pref_audio": pref_audio,
            "rej_audio": rej_audio,
        }
    
    def audio_encoding(self, batch):

        self.encoder.to(self.DEVICE)
        with torch.no_grad():
            pref_audio_inputs = self.encoder(torch.tensor(batch["pref_audio"]).to(self.DEVICE)).audio_codes
            rej_audio_inputs = self.encoder(torch.tensor(batch["rej_audio"]).to(self.DEVICE)).audio_codes
        
        # pad audio tokens
        pref_audio_inputs = F.pad(pref_audio_inputs, (0, 1280 - pref_audio_inputs.shape[-1]))
        rej_audio_inputs = F.pad(rej_audio_inputs, (0, 1280 - rej_audio_inputs.shape[-1]))

        # tokenize input text
        text_tokens = self.processor.tokenizer(
            text=batch["input_text"],
            padding="max_length",
            max_length=24,
            truncation=True,
            return_tensors="pt",
        )

        print(pref_audio_inputs.shape, rej_audio_inputs.shape, text_tokens.input_ids.shape)


        
        return {
            "input_ids": text_tokens.input_ids,
            "attention_mask": text_tokens.attention_mask,
            "pref_audio_tokens": pref_audio_inputs.squeeze(0).cpu(),
            "rej_audio_tokens": rej_audio_inputs.squeeze(0).cpu(),
        }

    def get_data_loader(self):
        print("Audio Loading...")
        # check if tokenized outcome exists
        if glob.glob('./tokenized_outcome/*'):
            print("Found tokenized outcome!")
            input_ids = torch.load('./tokenized_outcome/input_ids.pt')
            attention_mask = torch.load('./tokenized_outcome/attention_mask.pt')
            pref_audio_inputs = torch.load('./tokenized_outcome/pref_audio.pt')
            rej_audio_inputs = torch.load('./tokenized_outcome/rej_audio.pt')
            print("Data Shape:", input_ids.shape, attention_mask.shape, pref_audio_inputs.shape, rej_audio_inputs.shape)
            self.dataset = Dataset.from_dict({'input_ids': input_ids, 'attention_mask': attention_mask, 'pref_audio_tokens': pref_audio_inputs, 'rej_audio_tokens': rej_audio_inputs})
        
        else:
            dataset = self.dataset.map(self.feature_preparing, batched=True, num_proc=1, remove_columns=self.dataset.column_names, batch_size=64)

            print("Audio Encoding...")
            self.dataset = dataset.map(self.audio_encoding, batched=True, num_proc=1, remove_columns=dataset.column_names, batch_size=16)
            # save tokenized outcome
            if not os.path.exists('./tokenized_outcome'):
                os.makedirs('./tokenized_outcome')
            torch.save(torch.tensor(self.dataset['pref_audio_tokens']), './tokenized_outcome/pref_audio_test.pt')
            torch.save(torch.tensor(self.dataset['rej_audio_tokens']), './tokenized_outcome/rej_audio_test.pt')
            torch.save(torch.tensor(self.dataset['input_ids']), './tokenized_outcome/input_ids_test.pt')
            torch.save(torch.tensor(self.dataset['attention_mask']), './tokenized_outcome/attention_mask_test.pt')

        
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=default_data_collator)
        return data_loader


    def train(self):
        data_loader = self.get_data_loader()
        self.model.train()
        self.ref_model.eval()
        for epoch in range(self.args.epochs):
            losses = 0
            pbar = tqdm(data_loader, dynamic_ncols=True)
            self.optimizer.zero_grad()
            for time_step, batch in enumerate(pbar):
                text = batch['input_ids'].to(self.DEVICE)
                attn_mask = batch['attention_mask'].to(self.DEVICE)
                decoder_input_ids = (torch.ones((text.shape[0] * self.model.decoder.num_codebooks, self.args.max_output_length), dtype=torch.long)
                                     * self.model.generation_config.pad_token_id
                                ).to(self.DEVICE)
                pref_audio = batch['pref_audio_tokens'].to(self.DEVICE)
                rej_audio = batch['rej_audio_tokens'].to(self.DEVICE)
                model_audio = self.model(text, attn_mask, decoder_input_ids=decoder_input_ids, labels=pref_audio)
                ref_audio = self.ref_model(text, attn_mask, decoder_input_ids=decoder_input_ids, labels=pref_audio)
                model_pref_log_prob = get_log_prob(model_audio.logits, pref_audio)
                model_rej_log_prob = get_log_prob(model_audio.logits, rej_audio)
                ref_pref_log_prob = get_log_prob(ref_audio.logits, pref_audio)
                ref_rej_log_prob = get_log_prob(ref_audio.logits, rej_audio)
                loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = \
                    DPO_loss(model_pref_log_prob, model_rej_log_prob, ref_pref_log_prob, ref_rej_log_prob, beta=self.args.beta)
                loss /= self.args.accumulation_steps
                loss.backward()
                losses += loss.item() * self.args.accumulation_steps
                if time_step % self.args.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.args.use_wandb:
                    wandb.log({'loss': loss.item(),
                            'prefered_relative_logprob': prefered_relative_logprob,
                            'disprefered_relative_logprob': disprefered_relative_logprob,
                            'reward_accuracy': reward_accuracies,
                            'reward_margin': reward_margins})
                pbar.set_description(f"Step {time_step} | Rolling Loss: {loss.item():.4f}")
            pbar.close()
            print(f"Epoch {epoch + 1} Loss: {losses/len(data_loader)}")
            self.model.save_pretrained(f"{self.args.save_model_path}_epoch_{epoch+1}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_output_length", type=int, default=1280)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--encoder_path", type=str, default="facebook/encodec_32khz")
    parser.add_argument("--processor_path", type=str, default="facebook/musicgen-medium")
    parser.add_argument("--model_path", type=str, default="facebook/musicgen-medium")
    parser.add_argument("--save_model_path", type=str, default="./checkpoints/musicgen-dpo")
    parser.add_argument("--seed", type=int, default=1225)
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="musicgen-dpo")
    parser.add_argument("--wandb_run_name", type=str, default="musicgen-dpo-medium")


    args = parser.parse_args()

    seed_everything(seed=args.seed)

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(project=args.wandb_project, config=args, name=args.wandb_run_name)
    if not glob.glob("./tokenized_outcome/*"):
        print("Loading data...")
        pair_data = get_text_audio('./train_segments_25sec', './output_wav','music_label_genre.csv')
        print(f"Dataset Size:\nPreference Audio: {len(pair_data['pref_audio'])}\nRejected Audio: {len(pair_data['rej_audio'])}\nPrompt Text: {len(pair_data['text'])}")
        dataset = Dataset.from_dict(pair_data)
        print("Data loaded.")
    else:
        dataset = None

    trainer = DPOTrainer(args, dataset)
    trainer.load_model()
    trainer.train()

if __name__ == "__main__":
    main()