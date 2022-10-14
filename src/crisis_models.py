import os
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import DenseNet201
from keras.layers import Dense
from keras.models import Model
from crisis import CRISIS_MODEL_DIR
from crisis_datasets import load_crisis_dataset
from tensorflow.keras.optimizers import  Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ReduceLROnPlateau

from crisis_datasets import Dataset, load_crisis_dataset

# All the different models here

def vgg_model(num_class):
    vgg16 = VGG16(weights='imagenet')
    fc2 = vgg16.get_layer('fc2').output
    prediction = Dense(num_class, activation='softmax', name='predictions')(fc2)
    model = Model(inputs=vgg16.input, outputs=prediction)
    
    for layer in model.layers:
        if layer.name in ['predictions']:
            continue
        layer.trainable = False
        
        
    #print(model.summary())
    # Compile with SGD Optimizer and a Small Learning Rate
    #adam = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    #sgd = SGD(lr=1e-4, momentum=0.9)
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model(model_name, class_num):
    
    inputs = keras.Input(shape=(224, 224, 3))
  
    if model_name == 'resnet50':
        model = keras.applications.ResNet50(weights="imagenet", input_tensor=inputs, include_top=False)
        #return resnet50_model(class_num)
    elif model_name == 'vgg16':
        return vgg_model(class_num)
        model = keras.applications.VGG16(weights="imagenet", input_tensor=inputs, include_top=False)
    elif model_name == 'vgg162':
        return vgg_model(class_num)
        #model = keras.applications.VGG16(weights="imagenet", input_tensor=inputs, include_top=False)
    elif model_name == 'inceptionv3':
        model = keras.applications.InceptionV3(weights="imagenet", input_tensor=inputs, include_top=False)
        #return inceptionv3_model(class_num)
    elif model_name == 'mobilenetv2':
        model = keras.applications.MobileNetV2(weights="imagenet", input_tensor=inputs, include_top=False)
        #return mobilenetv2_model(class_num)
    elif model_name == 'densenet':
        model = keras.applications.DenseNet201(weights="imagenet", input_tensor=inputs, include_top=False)
    elif model_name == 'efficientnet':
        model = keras.applications.EfficientNetB0(weights="imagenet", input_tensor=inputs,include_top=False)
    elif model_name == 'efficientnetB7':
        model = keras.applications.EfficientNetB7(weights="imagenet", input_tensor=inputs,include_top=False)
    
    model.trainable = False

    x = model(inputs,training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(class_num, activation='softmax')(x)
    model = keras.Model(inputs,outputs)
    print(model.summary())
    return model

#def densenet_model(num_class):
#    
#    model_input = tf.keras.Input(shape=input_shape, name='input_layer')
#    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)    
#    outputs = []    
#
#    
#    x = DenseNet201(include_top=False, weights='imagenet', 
#                    input_shape=(224,224,3), 
#                    pooling='max')(dummy)
#
#    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#    outputs.append(x)
    
    model = tf.keras.Model(model_input, outputs, name='transfNetwork')
    model.summary()
    
    return model

model_makers={"vgg16":vgg_model}


# Main functions to work with models

def make_model(model_name, num_classes):
    return load_model(model_name, num_classes)
    return model_makers[model_name](num_classes)

def weights_filepath(dataset_name, model_name):
    filename=dataset_name+"."+model_name+".ckpt"
    return os.path.join(CRISIS_MODEL_DIR, filename)

def load_model_crisis(model_name, dataset):
    #model = make_model(model_name, dataset.num_classes)
    #model.load_weights(weights_filepath(dataset.name, model_name))
    #adam = Adam()
    #model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model = keras.models.load_model(weights_filepath(dataset.name, model_name))
    model._name = model_name
    return model

def load_model_multiple_crisis(model_name, dataset):
    #model = make_model(model_name, dataset.num_classes)
    #model.load_weights(weights_filepath(dataset.name, model_name))
    #adam = Adam()
    #model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model = keras.models.load_model(weights_filepath(dataset.name, model_name))
    model._name = dataset.name
    return model
#, {"accuracy":VALIDATION_ACCURACY, "loss":VALIDATION_LOSS}

# Training functions

def make_idg():
    def pre(x):
        return preprocess_input(x, mode='torch')
    return ImageDataGenerator(preprocessing_function=pre)

def train_model_crisis(model_name, dataset_name, enforce=False):
    
    dataset = load_crisis_dataset(dataset_name)
    if Path(weights_filepath(dataset.name, model_name)).exists() and not enforce:
        print(model_name + " has already been trained in dataset " + dataset_name)
        return 
    print("Starting training of model " + model_name + " in dataset " + dataset_name)
    idg = make_idg()
    train_data_generator = idg.flow_from_dataframe(
        dataset.training_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = True)

    valid_data_generator  = idg.flow_from_dataframe(
        dataset.dev_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = True)

    # Build the model 
    model = make_model(model_name, dataset.num_classes)
    adam = Adam()

    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    patience_learning_rate = 2
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=patience_learning_rate, verbose=1,
                                            factor=0.5, min_lr=0.00001,mode='min')
    
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=weights_filepath(dataset.name,model_name), mode='min', monitor='val_loss', verbose=2, save_best_only=True)
    num_epochs=50
    callbacks=[learning_rate_reduction, checkpoint]
    
    history = model.fit(x=train_data_generator,
                        epochs=num_epochs,
                        validation_data=valid_data_generator,
                        callbacks=callbacks)
    tf.keras.backend.clear_session()

    
def evaluate_model_crisis(m:Model, dataset:Dataset, idg):
    test_data_generator  = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = True)
    return m.evaluate(x=test_data_generator)
    
def load_and_evaluate_model_crisis(model_name, dataset_name):
    dataset = load_crisis_dataset(dataset_name)
    idg = make_idg()
    m = load_model_crisis(model_name, dataset)
    metrics = evaluate_model_crisis(m, dataset, idg)
    tf.keras.backend.clear_session()
    return metrics

def ensemble(model_names, dataset):
    # Load the models
    models=[load_model_crisis(model_name, dataset) for model_name in model_names]
    #for model in models:
    #    print(model.summary())
    input_shape = (224,224,3)
    input_img = keras.Input(shape=input_shape)

    outputs = [model(input_img) for model in models]
    y = keras.layers.Average()(outputs)

    model = Model(inputs=input_img, outputs=y, name='ensemble'.join(model_names))
    return model

def ensemble_disaster(model_name, datasets, dataset_name):
    # Load the models
    models=[load_model_multiple_crisis(model_name, dataset) for dataset in datasets]
    print(models)
    input_shape = (224,224,3)
    input_img = keras.Input(shape=input_shape)
    
    outputs = [model(input_img) for model in models]
    y = keras.layers.Average()(outputs)

    model = Model(inputs=input_img, outputs=y, name='ens'.join(dataset_name))
    return model

def ensemble_disaster_mix(model_name, datasets, dataset_name):
    # Load the models
    #models=[load_model_multiple_crisis(model_name, dataset) for dataset in datasets]
    models = []
    for dataset in datasets:
        if dataset == 'mexico_iraq':
            models.append(load_model_multiple_crisis('vgg16',dataset))
        else:
            models.append(load_model_multiple_crisis(model_name, dataset))
        
    print(models)
    input_shape = (224,224,3)
    input_img = keras.Input(shape=input_shape)
    
    outputs = [model(input_img) for model in models]
    y = keras.layers.Average()(outputs)

    model = Model(inputs=input_img, outputs=y, name='ens'.join(dataset_name))
    return model

def make_prediction(dataset,idg,model_dataset):
    model_dataset = load_crisis_dataset(model_dataset)
    tested_model = load_model_crisis('densenet',model_dataset)
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    predictions = tested_model.predict(X_test)
    return predictions

def make_prediction_supermodel(model,dataset,idg):
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    predictions = model.predict(X_test)
    
    return predictions


def make_prediction_ensemble(model_names,model_dataset,dataset,idg):
    model_dataset = load_crisis_dataset(model_dataset)
    model = ensemble(model_names, model_dataset)
    
    X_test = idg.flow_from_dataframe(
        dataset.test_data, directory = dataset.image_dir,
        target_size=(224, 224),
        x_col = dataset.image_column, y_col = dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    predictions = model.predict(X_test)
    
    return predictions

def make_prediction_ensemble_disaster(model,tested_dataset,idg):
    X_test = idg.flow_from_dataframe(
        tested_dataset.test_data, directory = tested_dataset.image_dir,
        target_size=(224, 224),
        x_col = tested_dataset.image_column, y_col = tested_dataset.label_column,
        class_mode = "raw", shuffle = False)
    
    predictions = model.predict(X_test)
    
    return predictions