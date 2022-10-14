from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shutil

def sorted_result_notinf(pred,num_data):
    probs = [i[0] for i in pred]
    sorted_index = np.argsort(probs)
    n = num_data
    get_sorted_index = sorted_index[:n]
    print(get_sorted_index)
    probs.sort()
    print(probs[:n])
    
    return get_sorted_index,probs

def sorted_result_inf(pred,num_data):
    probs = [i[0] for i in pred]
    sorted_index = np.argsort(probs)
    m = num_data
    get_sorted_index = sorted_index[-m:]
    print(get_sorted_index)
    probs.sort()
    print(probs[-m:])
    probs = probs[-m:]
    return get_sorted_index, probs


def get_class_inf(probs,m):
    y_pred = np.zeros(shape=(len(probs[-m:]),0),dtype=int)
    for x in probs[-m:]:
        if(x > 0.5):
                x = 0
                y_pred = np.append(y_pred,x)

        elif(x < 0.5):
                x = 1
                y_pred = np.append(y_pred,x)

    return y_pred

def get_class(probs):
    y_pred = np.zeros(shape=(len(probs[:n]),0),dtype=int)
    for x in probs[:n]:
        if(x > 0.5):
                x = 0
                y_pred = np.append(y_pred,x)

        elif(x < 0.5):
                x = 1
                y_pred = np.append(y_pred,x)

    return y_pred

def get_true_class(get_sorted_index,y_test):
    y_true = np.zeros(shape=(len(get_sorted_index),0),dtype=int)
    for j in range(len(get_sorted_index)):
        i = get_sorted_index[j]
        y_true = np.append(y_true,y_test[i])

    return y_true

def get_index_different_prediction(get_sorted_index,y_pred,y_true,probs):
    diff_prediction = np.zeros(shape=(len(get_sorted_index),0),dtype=int)
    get_proba = np.zeros(shape=(len(get_sorted_index),0),dtype=int)

    for i in range(len(y_pred)):
        if(y_pred[i] != y_true[i]):
            print("Fire !", i)
            print(get_sorted_index[i])
            print(probs[i])
            diff_prediction = np.append(diff_prediction,get_sorted_index[i])
            get_proba = np.append(get_proba,i)

    return diff_prediction, get_proba

def get_index_same_prediction(get_sorted_index,y_pred,y_true,probs):
    same_prediction = np.zeros(shape=(len(get_sorted_index),0),dtype=int)
    get_proba = np.zeros(shape=(len(get_sorted_index),0),dtype=int)

    for i in range(len(y_pred)):
        if(y_pred[i] == y_true[i]):
            print("Fire same !", i)
            print(get_sorted_index[i])
            print(probs[i])
            same_prediction = np.append(same_prediction,get_sorted_index[i])
            get_proba = np.append(get_proba,i)

    return same_prediction, get_proba


def show_image_different_prediction(filename,diff_prediction,probs,get_proba):
    if len(diff_prediction) == 0:
        print("No different result detected")
    else:
        df = pd.read_csv(filename, sep='\t', header=0)
        l = 0
        for k in diff_prediction:
            image_url = df['image_path'].iloc[k]
            img = mpimg.imread('/home/hafiz/data/crisismmd/event/'+image_url)
            imgplot = plt.imshow(img)
            plt.show()
            print(image_url)
            print(probs[get_proba[l]])
            dst_dir = "image_classification"
            shutil.copy('/home/hafiz/data/crisismmd/event/'+image_url,dst_dir)
            l = l+1
            #dest = 'image_classification/'
            #plt.savefig(image_url)
            
def show_image_same_prediction(filename,same_prediction,probs,get_proba):
    if len(same_prediction) == 0:
        print("No same result detected")
    else:
        df = pd.read_csv(filename, sep='\t', header=0)
        l = 0
        for k in same_prediction:
            image_url = df['image_path'].iloc[k]
            img = mpimg.imread('/home/hafiz/data/crisismmd/event/'+image_url)
            imgplot = plt.imshow(img)
            plt.show()
            print(image_url)
            print(probs[get_proba[l]])
            dst_dir = "image_classification"
            shutil.copy('/home/hafiz/data/crisismmd/event/'+image_url,dst_dir)
            l = l+1
            #dest = 'image_classification/'
            #plt.savefig(image_url)


def plot_confusion(model,test_dataset,y_true_init):
    
    y_true = y_true_init
    
    probabilities = model.predict(test_dataset)
    #print(probabilities[:10])
    y_pred = probabilities > 0.5
    y_pred = np.argmax(y_pred, axis=-1)
    #print(y_pred[:10])
    
    
    font = {
    'family': 'Times New Roman',
    'size': 12
    }
    
    matplotlib.rc('font', **font)
    conf_mat = confusion_matrix(y_true, y_pred)

    ax = sns.heatmap(conf_mat, annot=True, cmap='Blues',fmt='g')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['Informative','Not-Informative'])
    ax.yaxis.set_ticklabels(['Informative','Not-Informative'])

    plt.show()