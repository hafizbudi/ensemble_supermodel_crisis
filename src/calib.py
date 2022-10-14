import sys
sys.path.append("../src")
import matplotlib.pyplot as plt
from model import CalibratableModelFactory, CalibratableModelMixin
from plotting import (plot_calibration_curve,
                      plot_calibration_details_for_models,
                      plot_fitted_calibrator, plot_sample,
                      plot_sample_predictions)
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.calibration import calibration_curve,CalibratedClassifierCV

from crisis_ensemble_exp import train_base_models, evaluate_base_models, evaluate_ensemble_models
from crisis_models import load_model_crisis, make_idg, make_model, load_model, model_makers,evaluate_model_crisis, ensemble
from crisis_datasets import Dataset, load_crisis_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import  Adam, SGD
import tensorflow as tf
from calibration import IsotonicCalibrator, SigmoidCalibrator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from tensorflow import keras
from keras.models import Model

def load_data(filename,data_type):
    df = pd.read_csv(filename,sep="\t")
    
    if(data_type == 'individual'):
        df.image_info = pd.Categorical(df.image_info)
        df['label_image_code'] = df.image_info.cat.codes
        df[['image_path','image_info','label_image_code']]
    
    elif(data_type == 'whole'):
        df.label_image = pd.Categorical(df.label_image)
        df['label_image_code'] = df.label_image.cat.codes
        df[['image','label_image','label_image_code']]
   
    df.describe()
    
    return df

def pre_process_image(data,data_type):
    img_arr = []
    base_url = '/home/hafiz/Ensemble/ensemble_crisis/data/CrisisMMD_v2.0/'
    
    if data_type == 'individual':
        path = 'image_path'
    elif data_type == 'whole':
        path = 'image'
        
    for index,row in data.iterrows():
        read_img = cv2.imread(base_url+row[path])
        img_resize = cv2.resize(read_img,(224,224))
        img_ravel = img_resize.ravel()
        img_arr.append(img_ravel)
        img_np = np.array(img_arr)
        
    return img_np

def load_X_y(filename,data_type,dataset_name):
    
    data_test = load_data(filename,data_type)
    data_images_test = pre_process_image(data_test,data_type)
    dataset = load_crisis_dataset(dataset_name)
    
    X_test = data_images_test

    y_test = data_test['label_image_code'].values
    
    return X_test, y_test

def load_trained_model(dataset_name, model_name):
    
    factory = CalibratableModelFactory()
    dataset = load_crisis_dataset(dataset_name)
    model = load_model_crisis(model_name,dataset)
    #model = factory.get_model(load_model)

    return model,dataset

def calculate_predictions_brier(X_test,y_test,model):
    
    #predictions = model.predict(X_test.reshape(y_test.size,224,224,3))
    predictions = model.predict(X_test)
    treshold = 0.5
    
    #predictions_distribution = pd.Series(predictions[::2].round(3)).describe()
    #print("Predictions distribution: ", predictions_distribution)
    
    probs = [i[0] for i in predictions]


    #model_score = brier_score_loss(y_test,1-predictions[::2])
    model_score = brier_score_loss(y_test,probs)
    #return predictions, model_score
    return probs, model_score

def calculate_predictions_brier_ensemble(X_test,y_test,model):
    
    predictions = model.predict(X_test)
    treshold = 0.5
    
    probs = [i[0] for i in predictions]
    model_score = brier_score_loss(y_test,probs)
   
    return probs, model_score

def run_individual_calibration(filename,data_type,dataset_name,first_model_name,second_model_name):
    
    #load individual_model
    first_model,dataset = load_trained_model(dataset_name[0],first_model_name[0])
    second_model, dataset = load_trained_model(dataset_name[0],second_model_name[0])
    
    #load test_dataset
    idg = make_idg()
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    data_test = load_data(filename,data_type)
    y_test_inv = data_test['label_image_code'].values
    
    y_test = np.zeros(shape=(y_test_inv.size,0),dtype=int)
    for x in y_test_inv:
        if(x == 0):
            x = 1
            y_test = np.append(y_test,x)
    
        elif(x == 1):
            x = 0
            y_test = np.append(y_test,x)
            
    #compute predictions and brier score
    first_model_predictions, first_model_score = calculate_predictions_brier(X_test,y_test,first_model)
    second_model_predictions, second_model_score = calculate_predictions_brier(X_test,y_test,second_model)
    
    #visualize before calibration curve for individual
    show_plot_before_calibration(first_model_predictions,y_test,first_model_score,dataset_name,first_model_name)
    show_plot_before_calibration(second_model_predictions,y_test,second_model_score,dataset_name,second_model_name)
    
    #calibrate individual model
    method="isotonic"
#     first_model_calibration = individual_calibration(first_model,X_test,y_test,method,dataset_name,first_model_name)
#     second_model_calibration = individual_calibration(second_model,X_test,y_test,method,dataset_name,second_model_name)
    
    #1. run a classification model, load the existing model
    
    #2. use first_model_calibrated to calibrate the model
    
    #first_model_calibrated is not a model
    
    first_model_calibrator, first_model_brier_score = individual_calibration(first_model, X_test,y_test,method,dataset_name,first_model_name[0],first_model_predictions)
    
    second_model_calibrator, second_model_brier_score = individual_calibration(second_model, X_test, y_test,method,dataset_name,second_model_name[0],second_model_predictions)
    
    #ensemble two calibrated individual model
    ensemble_calibrated_model(first_model,first_model_calibrator,second_model,second_model_calibrator)

def ensemble_calibrated_model(first_model,first_model_calibrator,second_model,second_model_calibrator):
    
    print(callable(first_model))
    #create a list of ensemble model
    models = [first_model,second_model]
    
    input_shape = (224,224,3)
    input_img = keras.Input(shape=input_shape)

    outputs = [model(input_img) for model in models]
    y = keras.layers.Average()(outputs)
    
    #keras model or other way /regular python, input: images, output: probability calibration
    model = Model(inputs=input_img, outputs=y)
    return model


def individual_calibration(model, X_test, y_test,method,dataset_name,model_name,predictions):
    
    #generate prob_true and prob_pred from before calibration
    probs = predictions
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        
    #calibrate the model
    model = IsotonicRegression(out_of_bounds='clip').fit(prob_pred,prob_true)
    calibrated_predictions = model.predict(probs)
    
    calibrated_model_brier_score = brier_score_loss(y_test,calibrated_predictions)
    print(model_name,calibrated_model_brier_score)
    
    return model, calibrated_model_brier_score
    
    
def run_calibration(filename,data_type,dataset_name,model_name):
    
    #X_test, y_test = load_X_y(filename,data_type,dataset_name)
    model,dataset = load_trained_model(dataset_name,model_name)
    
    
    idg = make_idg()
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    data_test = load_data(filename,data_type)
    y_test_inv = data_test['label_image_code'].values
    #y_test = dataset.label_column
    #print("X_test",X_test[0][1])
    y_test = np.zeros(shape=(y_test_inv.size,0),dtype=int)
    for x in y_test_inv:
        if(x == 0):
            x = 1
            y_test = np.append(y_test,x)
    
        elif(x == 1):
            x = 0
            y_test = np.append(y_test,x)
    
 
    predictions, model_score = calculate_predictions_brier(X_test,y_test,model)
    show_plot_before_calibration(predictions,y_test,model_score,dataset_name,model_name)
    
    idg = make_idg()
    metrics = evaluate_model_crisis(model,dataset,idg)
    print(metrics)
    
    show_plot_after_calibration_isotonic(model, X_test, y_test,"isotonic",dataset_name,model_name,predictions)
    show_plot_after_calibration_sigmoid(model, X_test, y_test,"sigmoid",dataset_name,model_name,predictions)
    
    #show_plot_after_calibration(model, X_test, y_test,"isotonic",dataset_name,model_name)
    #show_plot_after_calibration(model, X_test, y_test,"sigmoid",dataset_name,model_name)

def run_calibration_ensemble(filename,data_type,dataset_name,model_names):
    
    dataset_name_text = dataset_name[0]
    dataset = load_crisis_dataset(dataset_name_text)
    model =  ensemble(model_names, dataset)
    
    idg = make_idg()
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    data_test = load_data(filename,data_type)
    y_test_inv = data_test['label_image_code'].values
    y_test = np.zeros(shape=(y_test_inv.size,0),dtype=int)
    
    for x in y_test_inv:
        if(x == 0):
            x = 1
            y_test = np.append(y_test,x)
    
        elif(x == 1):
            x = 0
            y_test = np.append(y_test,x)
            
    model_name = "ensemble"
    
    
    predictions, model_score = calculate_predictions_brier_ensemble(X_test,y_test,model)
    show_plot_before_calibration(predictions,y_test,model_score,dataset_name_text,model_name)
    
    #evaluate
    model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
    metrics = evaluate_model_crisis(model, dataset, idg)
    print("before ensemble calibration ",metrics)
    
    show_plot_after_calibration_isotonic(model, X_test, y_test,"isotonic",dataset_name_text,model_name,predictions)
    

def show_plot_before_calibration(predictions,y_test,model_score,dataset_name,model_name):
    
    probs = predictions
    
    prob_true, prob_pred = calibration_curve(y_true=y_test,y_prob=probs, 
                                                      n_bins=10)
    plt.figure(figsize=(10,10))

    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(prob_pred,prob_true, marker='.', label="%s (%0.3f)" % ('before calibration', model_score))
    ax1.legend()
    #ax1.set_title(dataset_name+' '+model_name)
    ax1.set_ylabel('fraction of positive')
    ax1.set_xlabel('probabilities')

    ax2.hist(probs, range=(0,1),bins=10, histtype="step",lw=2 )
    ax2.set_xlabel("mean predicted value")
    ax2.set_ylabel("count")
    
    #plt.savefig('/home/hafiz/Ensemble/ensemble_crisis/images/'+model_name+'_'+dataset_name+'.png')


def show_plot_after_calibration(model, X_test, y_test,method,dataset_name,model_name):
    X_test_predict = X_test.reshape(y_test.size,224,224,3)
   
   
    predictions = model.predict(X_test_predict)
    probs = predictions[::2]
    
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    if(method=="isotonic"):
        model.calibrators["isotonic"] = IsotonicCalibrator(prob_pred, prob_true)
    elif(method=="sigmoid"):
        model.calibrators["sigmoid"] = SigmoidCalibrator(prob_pred, prob_true)
    
    
    calibrated_predictions = model.predict_calibrated(X_test_predict,method)
    probs_calibrated = calibrated_predictions[::2]
    
    calibrated_model_score = brier_score_loss(y_test,probs_calibrated)
    
        
    prob_true = {}
    prob_pred = {}

    for i in range(y_test.shape[0]):
        prob_true[i], prob_pred[i] = calibration_curve(y_true=y_test,y_prob=probs_calibrated, 
                                                      n_bins=10)
    
    plt.figure(figsize=(10,10))

    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(prob_pred[0],prob_true[0], marker='.', label="%s (%0.3f)" % ('after calibration', calibrated_model_score))
    ax1.legend()
    #ax1.set_title(dataset_name+' '+model_name+' '+method)
    ax1.set_ylabel('fraction of positive')
    ax1.set_xlabel('probabilities')

    ax2.hist(probs_calibrated, range=(0,1),bins=10, histtype="step",lw=2 )
    ax2.set_xlabel("mean predicted value")
    ax2.set_ylabel("count")
    
    #plt.savefig('/home/hafiz/Ensemble/ensemble_crisis/images/'+model_name+'_'+dataset_name+'_'+method+'.png')
    

def show_plot_after_calibration_isotonic(model, X_test, y_test,method,dataset_name,model_name,predictions):
    
    #getmodel and generate prob_true and prob_pred from before calibration
    probs = predictions
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    
    #calibrate the model
    iso_reg = IsotonicRegression(out_of_bounds='clip').fit(prob_pred,prob_true)
    calibrated_predictions = iso_reg.predict(probs)
    calibrated_model_score = brier_score_loss(y_test,calibrated_predictions)
    
    #show brier score
    #print(calibrated_model_score)
    
    #show reliability curve
    prob_true, prob_pred = calibration_curve(y_true=y_test,y_prob=calibrated_predictions, 
                                                      n_bins=10)
    
    plt.figure(figsize=(10,10))

    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(prob_pred,prob_true, marker='.', label="%s (%0.3f)" % ('after calibration', calibrated_model_score))
    ax1.legend()
    ax1.set_title(dataset_name+' '+model_name+' '+method)
    ax1.set_ylabel('fraction of positive')
    ax1.set_xlabel('probabilities')

    ax2.hist(calibrated_predictions, range=(0,1),bins=10, histtype="step",lw=2 )
    ax2.set_xlabel("mean predicted value")
    ax2.set_ylabel("count")
    
    plt.savefig('/home/hafiz/Ensemble/ensemble_crisis/images/'+model_name+'_'+dataset_name+'_'+method+'.png')
    
def show_plot_after_calibration_sigmoid(model, X_test, y_test,method,dataset_name,model_name,predictions):
    
    #getmodel and generate prob_true and prob_pred of before calibration
    probs = np.array(predictions)
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    print("prob_true",prob_true)
    print("prob_pred",prob_pred)
    
    #log fraction positive and sigmoid function implementation
    #prob_pred, prob_true = _filter_out_of_domain(prob_pred, prob_true)
    prob_true = np.log(prob_true / (1 - prob_true))
    print("prob true log",prob_true)
    
    #calibrate the model
    sig =  LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1))
    
    #sig =  LinearRegression().fit(prob_pred, prob_true)
    #calibrated_predictions = sig.predict(probs.reshape(-1,1))
    
    #calibrated_predictions = 1 / (1 + np.exp(-sig.predict(probs.reshape(-1, 1)).flatten()))
    #calibrated_predictions = 1 / (1 + np.exp(-sig.predict(probs)))
    calibrated_predictions = 1 / (1 + np.exp(-sig.predict(probs.reshape(-1, 1))))
    
    calibrated_model_score = brier_score_loss(y_test,calibrated_predictions) 
    
    #show brier score
    print(calibrated_model_score)
    
    #show reliability curve
    prob_true, prob_pred = calibration_curve(y_true=y_test,y_prob=calibrated_predictions, 
                                                      n_bins=10)
    
    plt.figure(figsize=(10,10))

    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.plot(prob_pred,prob_true, marker='.', label="%s (%0.3f)" % ('after calibration', calibrated_model_score))
    ax1.legend()
    ax1.set_title(dataset_name+' '+model_name+' '+method)
    ax1.set_ylabel('fraction of positive')
    ax1.set_xlabel('probabilities')

    ax2.hist(calibrated_predictions, range=(0,1),bins=10, histtype="step",lw=2 )
    ax2.set_xlabel("mean predicted value")
    ax2.set_ylabel("count")

def _filter_out_of_domain(prob_pred, prob_true):
    filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
    return np.array(filtered)