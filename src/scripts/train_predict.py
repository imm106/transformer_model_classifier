import argparse
import os
import json

from sklearn.model_selection import train_test_split
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb


from src.common.loadData import load_all_data
from src.common.score import scorePredict
from src.model.classification_model import ClassificationModel
from src.common.utils import set_seed



load_dotenv(".env")

def main(parser):
    args = parser.parse_args()
    model_dir = args.model_dir
    label_to_exclude = args.label_to_exclude
    best_result_config = None
    config_path = args.config_path
    model_arg = args.model_arg

    with open(os.getcwd() + config_path) as f:
        general_args = json.load(f)
        
    df_model_args = pd.read_json(os.getcwd() + model_arg)
    training_args = df_model_args.to_dict(orient='records')[0]
    model_name = general_args["model_name"]

    set_seed(training_args["seed"])

    # Check if the model directory is provided or mantain the default Hugging Face model name 
    if model_dir != "":
        model_name = os.getcwd() + model_dir
        print(model_name)

    df_train=  load_all_data(general_args["train_file"], label_to_exclude, general_args["label"],
                                                  general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                                name_text_columns =general_args["name_text_columns"])
    labels = list(df_train['labels'].unique())
        
    if training_args["do_train"]:
        training_args["output_dir"] = os.path.join("models", general_args["output_dir"]) 
        
        # Configure W&B if the project name is provided
        if "WANDB_PROJECT" in general_args:
            os.environ["WANDB_PROJECT"] = general_args["WANDB_PROJECT"]  # name your W&B project
            os.environ["WANDB_LOG_MODEL"] = general_args["WANDB_LOG_MODEL"]  # log all model checkpoints
            wandb.login()
            #training_args.update(wandb.config)

        if training_args["do_eval"]:
            if general_args["eval_file"] != "":
                df_eval = load_all_data(general_args["eval_file"], label_to_exclude, general_args["label"],
                                                        general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                                        name_text_columns =general_args["name_text_columns"])
            else:
                train_size = 1- general_args["eval_size"]
                df_train, df_eval = train_test_split(df_train, test_size=general_args["eval_size"], train_size=train_size, random_state=1)
        
        if training_args["do_predict"]:
            if general_args["test_file"] != "":
                df_test = load_all_data(general_args["test_file"], label_to_exclude, general_args["label"],
                                                general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                                name_text_columns =general_args["name_text_columns"])
            else:
                train_size = 1- general_args["test_size"]
                df_train, df_test = train_test_split(df_train, test_size=general_args["test_size"], train_size=train_size, random_state=1)
        
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_eval) if training_args["do_eval"] else None,
            "test": Dataset.from_pandas(df_test) if training_args["do_predict"] else None
        })
        
        if "class_weights" in general_args:
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(df_train['labels'].values),
                                           y=df_train['labels'].values)

            # print("Clases únicas:", np.unique(df_train['labels'].values))
            # print("Peso por clase:", weights)
            
            general_args["class_weights"] = weights.tolist()
        
        model = ClassificationModel(model_name, training_args, dataset_dict, general_args)
        model.train()
    else:
        df_test = load_all_data(general_args["test_file"], label_to_exclude, general_args["label"],
                                                general_args["filter_label"], general_args["filter_label_value"], feature_name=general_args["features_name"],
                                                name_text_columns =general_args["name_text_columns"])
        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "test": Dataset.from_pandas(df_test)
        })
        training_args["do_train"] = False
        training_args["do_eval"] = False
        training_args["eval_strategy"]="no"
        model = ClassificationModel(model_name, training_args, dataset_dict, general_args)
        
    if training_args["do_predict"]:
        print("Predicting...")

        preds = model.predict()
        df_pred = pd.DataFrame(preds, columns=['labels'])
        
        # Add predictions to the test DataFrame and dump it to a CSV file
        df_test['pred'] = preds
        df_test.drop(columns=['features'], inplace=True, errors='ignore')
        parent_dir = os.path.dirname(os.getcwd() + general_args["test_file"])
        df_test.to_csv( parent_dir + "/predictions.csv", index=False)
        
        # Calculate the score
        labels_test = pd.Series(df_test['labels']).to_numpy()
        labels = list(df_test['labels'].unique())
        labels.sort()
        result, f1 = scorePredict(labels_test, df_pred.values, labels)
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config_path",
                        default="/config/constructividad/comment.json",
                        type=str,
                        help="File path to configuration parameters.")
    
    parser.add_argument("--model_arg",
                        default="/config/constructividad/comment_model.json",
                        type=str,
                        help="File path to model configuration parameters.")

    parser.add_argument("--model_dir",
                        default="",
                        type=str,
                        help="This parameter is the relative dir of model for predict.")

    parser.add_argument('--label_to_exclude',
                        default=[""],
                        nargs='+',
                        help="This parameter should be used if you want to execute experiments with fewer classes.")

    main(parser)
