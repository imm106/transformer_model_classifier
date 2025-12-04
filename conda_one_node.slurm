#!/bin/bash
#SBATCH --job-name=fine_tuning
#SBATCH --output=%j.out         # Nombre del archivo de salida
#SBATCH --error=%j.err          # Nombre del archivo de error
#SBATCH --partition=postiguet1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

# Activar entorno
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fine-tuning
nvidia-smi

# Cargar variables de entorno desde .env
set -a
source .env
set +a


huggingface-cli login --token $HF_TOKEN

#python -m src.scripts.train_predict --config_path /config/simplificacion/comment.json --model_arg /config/simplificacion/comment_model.json
#python -m src.scripts.train_predict --config_path /config/fraude/comment.json --model_arg /config/fraude/comment_model.json --model_dir="/models/comment_fraude_sms_1/checkpoint-1076/"
python -m src.scripts.upload_hf
#python -m src.scripts.train_predict --config_path /config/simplificacion/comment.json --model_arg /config/simplificacion/comment_model.json --model_dir="/models/comment_simplificacion_3/checkpoint-2198/"

#wandb agent rsepulveda/transformer_model_classifier/jigh6s0r