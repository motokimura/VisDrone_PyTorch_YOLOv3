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

### 2. Edit a bad format annotation file

```
$ nano VisDrone2018-DET-train/annotations/9999974_00000_d_0000053.txt
```

Edit this `VisDrone2018-DET-train/annotations/9999974_00000_d_0000053.txt`

from:

```
440,541,271,152,1,6,0,0,
436,772,134,72,1,5,0,0,
601,713,60,39,1,10,0,0,
486,706,102,59,1,8,0,0,
397,978,167,90,1,4,0,0,
718,1063,23,14,1,1,1,0,
696,13,21,33,1,3,0,0,
1124,73,98,62,1,5,0,1,
1089,35,84,46,1,7,0,1,
1351,5,48,26,1,10,0,1,
1381,28,60,72,1,8,0,0,
1370,113,44,37,1,10,0,0,
1272,143,91,55,1,7,0,0,
1323,128,46,31,1,10,0,1,
1300,135,33,24,1,3,0,1,
1260,246,83,39,1,7,0,0,
1266,211,90,40,1,7,0,1,
1225,100,13,36,1,3,0,0,
1070,159,200,104,1,6,0,0,
1072,283,139,64,1,5,0,0,
1052,371,138,79,1,5,0,0,
1046,441,141,93,1,5,0,0,
1294,316,84,51,1,7,0,0,
1291,367,61,39,1,8,0,0,
1026,553,145,76,1,4,0,0,
1043,620,119,78,1,5,0,0,
1216,608,94,56,1,7,0,0,
1240,682,52,31,1,10,0,0,
1232,745,92,45,1,7,0,1,
1414,493,22,38,1,1,0,0,
1438,482,76,45,1,7,0,0,
1446,673,88,60,1,7,0,0,
1460,242,62,146,1,6,0,0,
1098,0,175,38,1,6,1,0,
```

to:

```
440,541,271,152,1,6,0,0
436,772,134,72,1,5,0,0
601,713,60,39,1,10,0,0
486,706,102,59,1,8,0,0
397,978,167,90,1,4,0,0
718,1063,23,14,1,1,1,0
696,13,21,33,1,3,0,0
1124,73,98,62,1,5,0,1
1089,35,84,46,1,7,0,1
1351,5,48,26,1,10,0,1
1381,28,60,72,1,8,0,0
1370,113,44,37,1,10,0,0
1272,143,91,55,1,7,0,0
1323,128,46,31,1,10,0,1
1300,135,33,24,1,3,0,1
1260,246,83,39,1,7,0,0
1266,211,90,40,1,7,0,1
1225,100,13,36,1,3,0,0
1070,159,200,104,1,6,0,0
1072,283,139,64,1,5,0,0
1052,371,138,79,1,5,0,0
1046,441,141,93,1,5,0,0
1294,316,84,51,1,7,0,0
1291,367,61,39,1,8,0,0
1026,553,145,76,1,4,0,0
1043,620,119,78,1,5,0,0
1216,608,94,56,1,7,0,0
1240,682,52,31,1,10,0,0
1232,745,92,45,1,7,0,1
1414,493,22,38,1,1,0,0
1438,482,76,45,1,7,0,0
1446,673,88,60,1,7,0,0
1460,242,62,146,1,6,0,0
1098,0,175,38,1,6,1,0
```

by removing `,` at the end of each line.

### 3. Convert annotation files into a json file in MSCOCO format

```
$ python convert_labels2json.py
```
