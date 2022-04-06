import os
import glob
import json
import shutil
import collections
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import subprocess
import sys
import logging
import pickle
import boto3
import botocore

import torch.optim.optimizer
from datasets import load_metric

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification
from transformers.trainer_utils import IntervalStrategy


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def keep_best_model():
    """Find best saved checkpoint and delete remaining ones"""
    checkpoints = glob.glob(os.path.join(path_to_output, "checkpoint-*"))
    last_checkpoint = max([int(checkpoint.split("-")[-1]) for checkpoint in checkpoints])
    trainer_state = json.load(
        open(os.path.join(path_to_output, f"checkpoint-{last_checkpoint}", "trainer_state.json"), "r"))
    log_history = trainer_state["log_history"]
    eval_logs = [log for log in log_history if "eval_accuracy" in log]
    best_step = -1
    best_f1 = 0.0
    for log in eval_logs:
        best_step = log["step"]
        if log["eval_f1"] >= best_f1:
            best_f1 = log["eval_f1"]
        else:
            break
    for checkpoint in checkpoints:
        step = int(checkpoint.split("-")[-1])
        if step != best_step:
            shutil.rmtree(checkpoint)
        else:
            files = os.listdir(checkpoint)
            for file in files:
                shutil.move(os.path.join(path_to_output, checkpoint, file),
                            os.path.join(path_to_output, file))
            os.rmdir(checkpoint)


def convert_report_to_dataframe(report):
    lines = report.split("\n\n")
    df_columns = pd.DataFrame([x.split() for x in lines[0].split("\n")])
    columns = [df_columns.iloc[0, i] for i in range(df_columns.shape[1])]

    df = pd.DataFrame([x.split() for x in lines[1].split("\n")])
    df.index = df[0]
    df.drop(columns=[0], inplace=True)
    df.rename(columns={i + 1: columns[i] for i in range(len(columns))}, inplace=True)
    return df


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    pred_labels = [[id2label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in
                   zip(predictions, labels)]  # Remove ignored indexes (special tokens)
    true_labels = [[id2label[l] for (p, l) in zip(pred, label) if l != -100] for pred, label in
                   zip(predictions, labels)]
    report = classification_report(y_pred=pred_labels, y_true=true_labels)
    results = metric.compute(predictions=pred_labels, references=true_labels)
    return dict(precision=results["overall_precision"], recall=results["overall_recall"], f1=results["overall_f1"],
                accuracy=results["overall_accuracy"], report=report)


if __name__ == "__main__":
    install("seqeval")
    from seqeval.metrics import classification_report

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("@@@@@ ------- Training NER model has started ------- @@@@@")

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--model_artifacts", type=str)
    parser.add_argument("--bucket", type=str)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--models_dir", type=str, default=os.environ["SM_CHANNEL_MODELS"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_DATA"])

    args, _ = parser.parse_known_args()

    model_name_or_path = args.model_id
    path_to_dataset = args.data_dir
    preprocess_output = json.load(open(os.path.join(args.test_dir, "ner_preprocessing_output.json"), "r"))

    # directories
    global path_to_output, id2label
    
    model_id_new = preprocess_output["model_id_new"]
    path_to_output = os.path.join(os.environ["SM_MODEL_DIR"], f"/model_{model_id_new:02d}")
    path_to_initial_model = os.path.join(args.models_dir, f"model_{args.model_id:02d}") if preprocess_output["update"] else None

    # Get all of these below features from preprocess json
    glassbox = json.load(open(os.path.join(path_to_initial_model, "glassbox.json"), "r")) if preprocess_output["update"] else None
    config = json.load(open(os.path.join(path_to_initial_model, "config.json"), "r")) if preprocess_output["update"] else None
    target_entities_bio = sorted(list(config["label2id"].keys())) \
        if preprocess_output["update"] else sorted(
        ["O"] + ["-".join(["B", ent]) for ent in preprocess_output["target_entities"]] + ["-".join(["I", ent]) for ent in preprocess_output["target_entities"]])
    label2id = config["label2id"] if preprocess_output["update"] else {label: i for i, label in enumerate(target_entities_bio)}
    id2label = config["id2label"] if preprocess_output["update"] else {i: label for i, label in enumerate(target_entities_bio)}
    config = config if preprocess_output["update"] else AutoConfig.from_pretrained(model_name_or_path, num_labels=len(label2id), label2id=label2id, id2label=id2label)

    tokenizer = AutoTokenizer.from_pretrained(path_to_initial_model, use_fast=True, add_prefix_space=True) \
        if preprocess_output["update"] else AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, add_prefix_space=True)

    model = AutoModelForTokenClassification.from_pretrained(path_to_initial_model, config=config) \
        if preprocess_output["update"] else AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config)
    
    train_dataset_file = open(args.training_dir + "/train_dataset.pkl",'rb')
    train_dataset = pickle.load(train_dataset_file)
    dev_dataset_file = open(args.test_dir + "/dev_dataset.pkl",'rb')
    dev_dataset = pickle.load(dev_dataset_file)
    test_dataset_file = open(args.test_dir + "/test_dataset.pkl",'rb')
    test_dataset = pickle.load(test_dataset_file)
    
    # Set hyper-parameters (optimizer, weight decay, learning rate, scheduler)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay_rate": args.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_training_steps = len(train_dataset) / args.train_batch_size * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    optimizers = optimizer, scheduler

    training_args = TrainingArguments(
        output_dir=path_to_output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        max_grad_norm=args.max_grad_norm,
        dataloader_drop_last=False,
        log_level="debug",
        evaluation_strategy=IntervalStrategy.EPOCH,  # Evaluate at every end of an epoch
        logging_strategy=IntervalStrategy.EPOCH,  # Log results at every end of an epoch
        save_strategy=IntervalStrategy.EPOCH,  # Save checkpoint at every end of an epoch
        metric_for_best_model="f1",
        load_best_model_at_end=True,  # load the best model when finished training
        push_to_hub=False,
    )

    # collator batches processed examples together while applying padding to make them all the same size.
    # Each batch is padded to the length of its longest example.
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        optimizers=optimizers,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    results_before = trainer.evaluate(eval_dataset=test_dataset)
    report_before = convert_report_to_dataframe(results_before["eval_report"])
    
    trainer.train()
    
    results_after = trainer.evaluate(eval_dataset=test_dataset)
    report_after = convert_report_to_dataframe(results_after["eval_report"])
    
    keep_best_model()

    # load and save the best model
    best_model = AutoModelForTokenClassification.from_pretrained(path_to_output, config=config)
    torch.save(best_model.state_dict(), os.path.join(path_to_output, "pytorch_model.bin"),
               _use_new_zipfile_serialization=False)

    logger.info("@@@@@ ------- Training NER model has ended ------- @@@@@")
    
    if preprocess_output["update"]:
        glassbox["timestamp"] = str(datetime.now())
        glassbox["model_id"] = model_id_new
        glassbox["trained_on"].append("conll-2012")
        glassbox["trained_on"] = list(set(glassbox["trained_on"]))
        glassbox["training_data_genres"].extend(preprocess_output["genre"])
        glassbox["training_data_genres"] = list(set(glassbox["training_data_genres"]))
        glassbox["training_data_size"]["train"] += len(train_dataset)
        glassbox["training_data_size"]["dev"] += len(dev_dataset)
        glassbox["training_data_size"]["test"] += len(test_dataset)
        glassbox["evaluation_results"].append(
            dict(
                test_file=os.path.join(path_to_dataset, "test.txt"),
                overall_before=dict(
                    precision=float(results_before["eval_precision"]),
                    recall=float(results_before["eval_recall"]),
                    f1=float(results_before["eval_f1"]),
                    accuracy=float(results_before["eval_accuracy"])
                ),
                overall_after=dict(
                    precision=float(results_after["eval_precision"]),
                    recall=float(results_after["eval_recall"]),
                    f1=float(results_after["eval_f1"]),
                    accuracy=float(results_after["eval_accuracy"])
                ),
                entity_before={
                    entity: {metric: report_before[metric][entity] for metric in ["precision", "recall", "f1-score"]} for
                    entity in preprocess_output["target_entities"]},
                entity_after={
                    entity: {metric: report_after[metric][entity] for metric in ["precision", "recall", "f1-score"]} for
                    entity in preprocess_output["target_entities"]}
            )
        )
        json.dump(glassbox, open(os.path.join(path_to_output, "glassbox.json"), "w"), indent=4)
    else:
        # create a glassbox object
        glassbox_obj = dict(
            timestamp=str(datetime.now()),
            model_id=model_id_new,
            trained_on=["conll-2012"],
            training_data_genres=preprocess_output["genre"],
            training_data_size=dict(
                train=len(train_dataset),
                dev=len(dev_dataset),
                test=len(test_dataset)
            ),
            entities=preprocess_output["target_entities"],
            evaluation_results=[
                dict(
                    test_file=os.path.join(path_to_dataset, "test.txt"),
                    overall_before=dict(
                        precision=float(results_before["eval_precision"]),
                        recall=float(results_before["eval_recall"]),
                        f1=float(results_before["eval_f1"]),
                        accuracy=float(results_before["eval_accuracy"])
                    ),
                    overall_after=dict(
                        precision=float(results_after["eval_precision"]),
                        recall=float(results_after["eval_recall"]),
                        f1=float(results_after["eval_f1"]),
                        accuracy=float(results_after["eval_accuracy"])
                    ),
                    entity_before={
                        entity: {metric: report_before[metric][entity] for metric in ["precision", "recall", "f1-score"]}
                        for entity in preprocess_output["target_entities"]},
                    entity_after={
                        entity: {metric: report_after[metric][entity] for metric in ["precision", "recall", "f1-score"]} for
                        entity in preprocess_output["target_entities"]}
                )
            ]
        )
        json.dump(glassbox_obj, open(os.path.join(path_to_output, "glassbox.json"), "w"), indent=4)
    
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])