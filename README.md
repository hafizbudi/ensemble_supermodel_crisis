# Single vs ensemble model : A comparative study for social media images classification in disaster response
Supermodel and ensemble model are two different approach that we use to assess the deep learning adaptability to unseen disaster.
The adaptability of deep learning model is critical in the event of disaster. Please kindly check our paper here

# To replicate the experiments

1. git clone this repo to the directory of your choice and cd there
2. `source bin/init_local.sh`
3. `source bin/download_dataset.sh`
4. `python bin/run_ensemble_crisis.py`. This will take one full day... Finally, it will create a file `metrics.csv` which contains the results of running the experiments. You can compare it with [metrics.paper.csv](nb/metrics.paper.csv) to see if there are significant differences. 

## Installation
Before executing the code, you need python version 3 

## Download dataset
You can download the CrisisMMD here (https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz)

## Install dependencies
```
pip install -r requirements.txt
```

## Please cite the following paper if you use the CrisisMMD data:

* *Ferda Ofli, Firoj Alam, and Muhammad Imran, "Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response" (https://arxiv.org/pdf/2004.11838.pdf), 17th International Conference on Information Systems for Crisis Response and Management, 2020.*

* *Firoj Alam, Ferda Ofli, and Muhammad Imran, "Crisismmd: Multimodal twitter datasets from natural disasters" (https://arxiv.org/pdf/1805.00713.pdf), Twelfth International AAAI Conference on Web and Social Media. 2018.*


```bib
@inproceedings{multimodalbaseline2020,
  Author = {Ferda Ofli and Firoj Alam and Muhammad Imran},
  Booktitle = {17th International Conference on Information Systems for Crisis Response and Management},
  Keywords = {Multimodal deep learning, Multimedia content, Natural disasters, Crisis Computing, Social media},
  Month = {May},
  Organization = {ISCRAM},
  Publisher = {ISCRAM},
  Title = {Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response},
  Year = {2020}
}
@inproceedings{crisismmd2018icwsm,
  author = {Firoj Alam and Ofli, Ferda and Imran, Muhammad},
  title = {CrisisMMD: Multimodal Twitter Datasets from Natural Disasters},
  booktitle = {Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)},
  year = {2018},
  month = {June},
  date = {23-28},
  location = {USA}}
```