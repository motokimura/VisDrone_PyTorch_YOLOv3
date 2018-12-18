# VisDrone2018-DET-py

This repository provides:

1. A python script to convert VisDrone2018-DET annotation files into a json file in MSCOCO format
2. Some jupyter notebooks to analyze VisDrone2018-DET dataset

## Dependency

Install dependencies by following:

```
$ pip install -r requirements.txt
```

## Usage

### 1. Download VisDrone2018-DET dataset

After downloading `VisDrone2018-DET-train.zip`, `VisDrone2018-DET-val.zip`, 
and `VisDrone2018-VID-test-challenge.zip` from [VisDrone website](http://aiskyeye.com/views/index)., 
place them in `data` directory and unzip.

```
$ cd visdrone_scripts
$ mkdir data

$ mv path/to/VisDrone2018-DET-train.zip data
$ mv path/to/VisDrone2018-DET-val.zip data
$ mv path/to/VisDrone2018-VID-test-challenge.zip data

$ cd data
$ unzip VisDrone2018-DET-train.zip
$ unzip VisDrone2018-DET-val.zip
$ unzip VisDrone2018-VID-test-challenge.zip
```

### 2. Convert annotation files into a json file in MSCOCO format

```
$ python convert_labels2json.py
```
