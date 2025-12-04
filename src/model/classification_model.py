from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn as nn
from typing import Union, Any, Optional
import torch


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.class_weights is not None:             
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
        else:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch) 
            if return_outputs:
                loss, outputs = loss
        return (loss, outputs) if return_outputs else loss


class ClassificationModel:
    def __init__(self, model_id, training_args, dataset_dict, general_args):
        print(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_encoder = None
        self.general_args = general_args

        
        self.class_weights = torch.Tensor(general_args["class_weights"])

        unique_labels = sorted(set(dataset_dict["train"]["labels"]))
        self.id2label = {i: str(l) for i, l in enumerate(unique_labels)}
        self.label2id = {str(l): i for i, l in enumerate(unique_labels)}


        config = AutoConfig.from_pretrained(
            model_id, 
            num_labels=len(unique_labels),
            id2label=self.id2label,
            label2id=self.label2id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        if self.label_encoder:
            config.num_labels = len(self.label_encoder.classes_)
        if "dropout_rate" in training_args:
            config.hidden_dropout_prob = training_args["dropout_rate"]
            config.attention_probs_dropout_prob = training_args["dropout_rate"]
            training_args.pop("dropout_rate")
         
        self.training_args = TrainingArguments(**training_args)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, config=config
        )
        
        self.tokenized_datasets = dataset_dict.map(
            self.preprocessing_function, batched=True
        )

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = None

    def label_encoding(self, examples):
        examples = [self.label2id[example["label"]] for example in examples]
        return examples

    """def label_encoding(self, examples):
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(examples["labels"])
        return self.label_encoder.transform(examples["labels"])
    """

    def preprocessing_function(self, examples):
        if "text_b" in examples:
            encoded = self.tokenizer(
                examples["text_a"],
                examples["text_b"],
                padding="max_length",
                max_length=self.general_args["max_length"],
                return_tensors="pt",
                truncation=True,
            )
        else:
            encoded = self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=self.general_args["max_length"],
                return_tensors="pt",
                truncation=True,
            )

        #encoded["labels"] = self.label_encoding(examples)
        encoded["labels"] = [self.label2id[label] for label in examples["labels"]]
        return encoded

    def train(self):
        self.trainer = WeightedLossTrainer(
            self.model,
            self.training_args,
            class_weights= self.class_weights,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
        )
        self.trainer.train()

    def predict(self):
        if self.trainer is None:
            self.trainer = Trainer(
                self.model,
                self.training_args,
                data_collator=self.data_collator,
                processing_class=self.tokenizer,
            )
        predictions = self.trainer.predict(self.tokenized_datasets["test"])
        preds = np.argmax(predictions.predictions, axis=-1)

        pred_labels = [self.id2label[i] for i in preds]


        return pred_labels

        #return self.label_encoder.inverse_transform(preds)
