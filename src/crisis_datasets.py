from dataclasses import dataclass
import pandas as pd
from crisis import CRISIS_DATA_DIR

@dataclass 
class Dataset:
    name:str
    num_classes:int
    label_column:str
    image_dir:str
    image_column:str
    training_data: pd.core.frame.DataFrame
    dev_data: pd.core.frame.DataFrame
    test_data: pd.core.frame.DataFrame

def tdt_loader(base_path,  train_tsv, dev_tsv, test_tsv, label_column):
    training_data = pd.read_csv(base_path / train_tsv, sep='\t')
    training_data[label_column], index = pd.factorize(training_data[label_column], sort=True)
    # training_data = training_data.iloc[0:100]
    dev_data = pd.read_csv(base_path / dev_tsv, sep='\t')
    dev_data[label_column] = pd.factorize(dev_data[label_column], sort=True)[0]
    # dev_data = dev_data.iloc[0:100]
    test_data = pd.read_csv(base_path / test_tsv, sep='\t')
    test_data[label_column] = pd.factorize(test_data[label_column], sort=True)[0]
    num_classes = index.size
    return training_data, dev_data, test_data, num_classes

def put_supermodel_data(datasets):
    for dataset in datasets:
        a = load_crisis_dataset(dataset)
        b = a.image_dir
        print(b)
    #save as tsv
    

def load_crisis_dataset(dataset_name=None, base_path=CRISIS_DATA_DIR):
    if dataset_name is None:
        dataset_name = "informative_agreed"
    label_column="label_image"
    image_dir = base_path / "CrisisMMD_v2.0"
    image_column="image"
    if dataset_name == "informative_agreed":
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_agreed_label",
                                                               "task_informative_text_img_agreed_lab_train.tsv",
                                                               "task_informative_text_img_agreed_lab_dev.tsv",
                                                               "task_informative_text_img_agreed_lab_test.tsv",
                                                               label_column)
    elif dataset_name == "humanitarian_agreed":
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_agreed_label",
                                                               "task_humanitarian_text_img_agreed_lab_train.tsv",
                                                               "task_humanitarian_text_img_agreed_lab_dev.tsv",
                                                               "task_humanitarian_text_img_agreed_lab_test.tsv",
                                                               label_column)
    elif dataset_name == "damage_all":
        label_column = "label"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all",
                                                               "task_damage_text_img_train.tsv",
                                                               "task_damage_text_img_dev.tsv",
                                                               "task_damage_text_img_test.tsv",
                                                               label_column)
    elif dataset_name == "humanitarian_all":
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all",
                                                               "task_humanitarian_text_img_train.tsv",
                                                               "task_humanitarian_text_img_dev.tsv",
                                                               "task_humanitarian_text_img_test.tsv",
                                                               label_column)
    elif dataset_name == "informative_all":
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all",
                                                               "task_informative_text_img_train.tsv",
                                                               "task_informative_text_img_dev.tsv",
                                                               "task_informative_text_img_test.tsv",
                                                               label_column)
    elif dataset_name == 'iraq_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_iraq_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'iraq_all_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_iraq_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'iraq_all_hum':
        label_column="image_human"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_iraq_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'iraq_all_test_mexico_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'iraq_all_test_srilanka_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'iraq_all_test_california_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'iraq_all_test_irma_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'iraq_all_test_harvey_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'iraq_all_test_maria_rerun':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_iraq_train.tsv",
                                                               "task_informative_iraq_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
        
        
    elif dataset_name == 'mexico_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_mexico_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'mexico_all_hum':
        label_column="image_human"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_mexico_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'mexico_all_test_iraq':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'mexico_all_test_srilanka':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'mexico_all_test_california':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'mexico_all_test_irma':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'mexico_all_test_harvey':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'mexico_all_test_maria':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_mexico_train.tsv",
                                                               "task_informative_mexico_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_srilanka_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_test_irma':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_test_iraq':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
    
    
    elif dataset_name == 'srilanka_all_test_mexico':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_test_california':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_test_harvey':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_test_maria':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'srilanka_all_hum':
        label_column="image_human"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_srilanka_train.tsv",
                                                               "task_informative_srilanka_dev.tsv",
                                                               "task_informative_srilanka_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'irma_all':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_irma_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'irma_all_test_iraq':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'irma_all_test_mexico':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'irma_all_test_srilanka':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'irma_all_test_california':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'irma_all_test_harvey':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'irma_all_test_maria':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)


    elif dataset_name == 'irma_all_hum':
        label_column='image_human'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_irma_train.tsv",
                                                               "task_informative_irma_dev.tsv",
                                                               "task_informative_irma_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'harvey_all':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_harvey_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'harvey_all_test_iraq':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'harvey_all_test_mexico':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'harvey_all_test_srilanka':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'harvey_all_test_california':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'harvey_all_test_irma':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'harvey_all_test_maria':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'harvey_all_hum':
        label_column='image_human'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_harvey_train.tsv",
                                                               "task_informative_harvey_dev.tsv",
                                                               "task_informative_harvey_test.tsv",
                                                               label_column)
    
     
    elif dataset_name == 'maria_all':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_maria_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_iraq':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_srilanka':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_california':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_california_wildfires_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_irma':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_harvey':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_maria':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_test_mexico':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'maria_all_hum':
        label_column='image_human'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_maria_train.tsv",
                                                               "task_informative_maria_dev.tsv",
                                                               "task_informative_maria_test.tsv",
                                                               label_column)
    
    
    elif dataset_name == 'california_all':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_california_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'california_all_test_iraq':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_iraq_iran_earthquake_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'california_all_test_mexico':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_mexico_earthquake_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'california_all_test_srilanka':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_srilanka_floods_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'california_all_test_irma':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_hurricane_irma_final_data.tsv",
                                                               label_column)
        
    elif dataset_name == 'california_all_test_harvey':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_hurricane_harvey_final_data.tsv",
                                                               label_column)
    elif dataset_name == 'california_all_test_maria':
        label_column='image_info'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_hurricane_maria_final_data.tsv",
                                                               label_column)
    
    elif dataset_name == 'california_all_hum':
        label_column='image_human'
        image_column='image_path'
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "task_informative_california_train.tsv",
                                                               "task_informative_california_dev.tsv",
                                                               "task_informative_california_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'without_irma_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_irma_train.tsv",
                                                               "without_irma_dev.tsv",
                                                               "without_irma_test.tsv",
                                                               label_column)
    elif dataset_name == 'without_california_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_california_train.tsv",
                                                               "without_california_dev.tsv",
                                                               "without_california_test.tsv",
                                                               label_column)
    elif dataset_name == 'without_harvey_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_harvey_train.tsv",
                                                               "without_harvey_dev.tsv",
                                                               "without_harvey_test.tsv",
                                                               label_column)
    elif dataset_name == 'without_srilanka_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_srilanka_train.tsv",
                                                               "without_srilanka_dev.tsv",
                                                               "without_srilanka_test.tsv",
                                                               label_column)
    elif dataset_name == 'without_iraq_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_iraq_train.tsv",
                                                               "without_iraq_dev.tsv",
                                                               "without_iraq_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'without_mexico_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_mexico_train.tsv",
                                                               "without_mexico_dev.tsv",
                                                               "without_mexico_test.tsv",
                                                               label_column)
        
    elif dataset_name == 'without_maria_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "without_maria_train.tsv",
                                                               "without_maria_dev.tsv",
                                                               "without_maria_test.tsv",
                                                               label_column)    
    elif dataset_name == 'supermodel_all':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "supermodel_all_train.tsv",
                                                               "supermodel_all_dev.tsv",
                                                               "supermodel_all_test.tsv",
                                                               label_column)
    elif dataset_name == 'hurricane_floods':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "hurricane_floods_train.tsv",
                                                               "hurricane_floods_dev.tsv",
                                                               "hurricane_floods_test.tsv",
                                                               label_column)
    elif dataset_name == 'hurricane_floods_without_irma':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "hurricane_floods_without_irma_train.tsv",
                                                               "hurricane_floods_without_irma_dev.tsv",
                                                               "hurricane_floods_without_irma_test.tsv",
                                                               label_column)
    elif dataset_name == 'hurricane_floods_without_maria':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "hurricane_floods_without_maria_train.tsv",
                                                               "hurricane_floods_without_maria_dev.tsv",
                                                               "hurricane_floods_without_maria_test.tsv",
                                                               label_column)
    elif dataset_name == 'hurricane_floods_without_harvey':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "hurricane_floods_without_harvey_train.tsv",
                                                               "hurricane_floods_without_harvey_dev.tsv",
                                                               "hurricane_floods_without_harvey_test.tsv",
                                                               label_column)
    elif dataset_name == 'hurricane_without_floods':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "hurricane_without_floods_train.tsv",
                                                               "hurricane_without_floods_dev.tsv",
                                                               "hurricane_without_floods_test.tsv",
                                                               label_column)
    
    elif dataset_name == 'earthquakes':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "earthquakes_train.tsv",
                                                               "earthquakes_dev.tsv",
                                                               "earthquakes_test.tsv",
                                                               label_column)
    elif dataset_name == 'harvey_maria':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "harvey_maria_train.tsv",
                                                               "harvey_maria_dev.tsv",
                                                               "harvey_maria_test.tsv",
                                                               label_column)
    elif dataset_name == 'mexico_iraq':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "mexico_iraq_train.tsv",
                                                               "mexico_iraq_dev.tsv",
                                                               "mexico_iraq_test.tsv",
                                                               label_column)
    elif dataset_name == 'irma_maria':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "irma_maria_train.tsv",
                                                               "irma_maria_dev.tsv",
                                                               "irma_maria_test.tsv",
                                                               label_column)
    elif dataset_name == 'irma_harvey':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "irma_harvey_train.tsv",
                                                               "irma_harvey_dev.tsv",
                                                               "irma_harvey_test.tsv",
                                                               label_column)
    elif dataset_name == 'harvey_maria_irma':
        label_column="image_info"
        image_column="image_path"
        training_data, dev_data, test_data, num_classes = tdt_loader(base_path / "crisismmd_datasplit_all_individual",
                                                               "harvey_maria_irma_train.tsv",
                                                               "harvey_maria_irma_dev.tsv",
                                                               "harvey_maria_irma_test.tsv",
                                                               label_column)   
    
    return Dataset(dataset_name, num_classes, label_column, image_dir, image_column, training_data, dev_data, test_data) 
    
    