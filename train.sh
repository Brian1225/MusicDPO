for lr in 0.001 0.0005; do
    python trainer.py --epochs 15 --lr $lr --batch_size 2 --accumulation_steps 16 --lora_r 16 --lora_alpha 16 --model_path facebook/musicgen-small --use_wandb --wandb_run_name musicgen-dpo-CLAP-$lr --save_model_path ./checkpoints-CLAP-$lr/musicgen-dpo
done    