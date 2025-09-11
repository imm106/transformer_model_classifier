from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv(".env")


api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(
    repo_id="gplsi/mRoBERTa_FT1_DFT1_fraude_phishing",
    repo_type="model",
    private=True,
    exist_ok=True  # won't fail if it already exists
)


api.upload_folder(
    folder_path="/home/gplsi/GPLSI/codigos/transformer_model_classifier/models/comment_fraude_phishing_1/checkpoint-2120",
    repo_id="gplsi/mRoBERTa_FT1_DFT1_fraude_phishing",
    repo_type="model",
)
