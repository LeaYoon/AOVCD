# Attribute-aware Open-Vocabulary Crack Detection
This repo provide guidance for port road data and codes for attribute-aware open-vocabulary crack detection.

## Requirements
- Python >= 3.10
- pytorch >= 2.1.0
We have tested our project with two NVIDIA RTX 6000 Ada Generation. 

## Installation
```bash
git clone https://github.com/LeaYoon/AOVCD.git
cd AOVCD
pip install -r requirements.txt
```


## Dataset
We analyze AOVCD model on [(PPDD Dataset)](https://github.com/LeaYoon/PPDD/blob/main/README.md). 

PPDD Dataset download link : [Click here to try](https://drive.google.com/drive/folders/1jiR-q0W8wZvoQqv-a1otfEKdToatf6lZ?usp=sharing)

To use PPDD data for our project, first you need to be processed into COCO-style. 
We will upload soon after publishing paper and arranging project!

## Train
```bash
sh train_ppdd_ovd.sh
```

## Train
```bash
sh test_ppdd_ovd.sh
```
