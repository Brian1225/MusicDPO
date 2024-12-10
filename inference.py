from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import LoraConfig, PeftModel
import torch
from tqdm import tqdm
import scipy
import os
import argparse

def main(args):
    lora_model = LoraConfig.from_pretrained(args.peft_model)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = MusicgenForConditionalGeneration.from_pretrained(args.base_model)
    ft_model = PeftModel(model, lora_model)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    inputs = processor(
        text=["Solo Violin in D major with Baroque style"],
        padding=True,
        return_tensors="pt",
    )
    inputs.to(DEVICE)
    ft_audio_values = ft_model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1280) # every 5 seconds will need 256 tokens
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1280) # every 5 seconds will need 256 tokens
    print(audio_values.shape)
    sampling_rate = model.config.audio_encoder.sampling_rate
    output_path = os.path.join(args.output_dir, "musicgen_out.wav")
    ft_output_path = os.path.join(args.output_dir, "musicgen_ft_out.wav")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=ft_audio_values.cpu().numpy())
    scipy.io.wavfile.write(ft_output_path, rate=sampling_rate, data=audio_values.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model", type=str, default="./checkpoints-30/musicgen-dpo_epoch_10/")
    parser.add_argument("--base_model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    main(args)


