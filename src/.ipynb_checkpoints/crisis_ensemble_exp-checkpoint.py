import itertools
import pandas as pd

from crisis_datasets import load_crisis_dataset
from crisis_models import train_model_crisis, load_and_evaluate_model_crisis, ensemble, make_idg, evaluate_model_crisis
import tensorflow as tf

#dataset_names = ["informative_agreed","humanitarian_agreed","damage_all","informative_all","humanitarian_all"]

#dataset_names = ["informative_agreed","informative_all"]
dataset_names = ["harvey_all","irma_all","maria_all","mexico_all","iraq_all","california_all","srilanka_all"]
#dataset_names = ["maria_all_test_iraq","maria_all_test_mexico","maria_all_test_srilanka","maria_all_test_california","maria_all_test_irma","maria_all_test_harvey"]

model_names = ["vgg16","densenet","mobilenetv2","inceptionv3","resnet50"]

#model_names = ["efficientnet","efficientnetB7"]

def train_base_models(dataset_names=dataset_names, model_names=model_names):
    for dataset_name in dataset_names:
        for model_name in model_names:
            print(model_name, dataset_name)
            train_model_crisis(model_name, dataset_name)

def save_metrics(df_metrics, basename="metrics"):
    df_metrics.to_csv(basename + ".csv", index=False)
    df_metrics.to_parquet(basename + ".parquet")
    
def load_metrics(basename="metrics"):
    try:
        df_metrics = pd.read_parquet(basename + ".parquet")
    except FileNotFoundError:
        try:
            df_metrics = pd.read_csv(basename + ".csv")
        except FileNotFoundError:
            df_metrics = pd.DataFrame(columns = ["dataset","model","loss","accuracy"])
    return df_metrics

def evaluate_base_models(df_metrics=None, dataset_names=dataset_names, model_names=model_names):
    if df_metrics is None:
        df_metrics = load_metrics()
    for dataset_name in dataset_names:
        for model_name in model_names:
            print("Dataset:", dataset_name, "Model:", model_name)
            try:
                if ((df_metrics['dataset'] == dataset_name) & (df_metrics['model'] == model_name)).any():
                    print("Already evaluated")
                    continue
                metrics = load_and_evaluate_model_crisis(model_name, dataset_name)
                df_metrics.loc[len(df_metrics)]=[dataset_name, model_name] + metrics
                print(metrics)                        
            except Exception as e:
                print(e)
                print("Model not yet trained")
    tf.keras.backend.clear_session()
    save_metrics(df_metrics)
    return df_metrics

# def evaluate_ensemble_models(df_metrics=None, dataset_names=dataset_names, model_names=model_names):
#     if df_metrics is None:
#         df_metrics = load_metrics()
#     idg = make_idg()
#     for dataset_name in dataset_names:
#         dataset = load_crisis_dataset(dataset_name)
#         for e in itertools.combinations(model_names, 2):
#             model_name = str(e)
#             print("Dataset:",dataset_name,"Model:",model_name)
#             try:
#                 if ((df_metrics['dataset'] == dataset_name) & (df_metrics['model'] == model_name)).any():
#                     print("Already evaluated")
#                     continue
#                 m = ensemble(e, dataset)
#                 m.compile(loss='sparse_categorical_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'])
#                 metrics = evaluate_model_crisis(m, dataset, idg)
#                 df_metrics.loc[len(df_metrics)]=[dataset_name, model_name] + metrics
#                 print(metrics)
#             except Exception as e:
#                 print(e)
#                 print("Model not yet trained")
#     tf.keras.backend.clear_session()
#     save_metrics(df_metrics)

def evaluate_ensemble_models(df_metrics=None, dataset_names=dataset_names, model_names=model_names):
    if df_metrics is None:
        df_metrics = load_metrics()
    idg = make_idg()
    for dataset_name in dataset_names:
        dataset = load_crisis_dataset(dataset_name)
        for ensemble_size in range(2,len(model_names)+1):
            for e in itertools.combinations(model_names, ensemble_size):
                model_name = str(e)
                print("Dataset:",dataset_name,"Model:",model_name)
                try:
                    if ((df_metrics['dataset'] == dataset_name) & (df_metrics['model'] == model_name)).any():
                        print("Already evaluated")
                        continue
                    m = ensemble(e, dataset)
                    m.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                    metrics = evaluate_model_crisis(m, dataset, idg)
                    df_metrics.loc[len(df_metrics)]=[dataset_name, model_name] + metrics
                    print(metrics)
                except Exception as e:
                    print(e)
                    print("Model not yet trained")
    tf.keras.backend.clear_session()
    save_metrics(df_metrics)