# EduRec: Disentangled Recommendation of Video Topics in MOOC

This is our TensorFlow implementation for the paper:

[Shalini Pandey](https://www.linkedin.com/in/shalini-pandey-91844958), [Abinash Sinha](https://www.linkedin.com/in/abinashsinha330),
[Jaideep Srivastava](https://cse.umn.edu/cs/jaideep-srivastava) (2021). *[EduRec: Disentangled Recommendation of Video Topics in MOOC.](https://github.com/abinashsinha330/EduRec)* '21

Please cite our paper if you use the code or datasets.

The code is tested under a Linux desktop with TensorFlow 2.5.0 and Python 3.6.7.


## Datasets

The preprocessed datasets are included in the repo (`e.g. data/MOOCCube.txt`), where each line contains an `student id` and 
`concept id` (starting from 1) meaning an interaction (sorted by timestamp).

The data pre-processing script is also included. For example, you could download MOOCCube data from *[here.](http://lfs.aminer.cn/misc/moocdata/data/MOOCCube.zip)*, and run the script to produce the `txt` format data.
  

## Model Training

To train our model on `MOOCCube` (with default hyper-parameters): 

```
python main.py --dataset=MOOCCube --train_dir=default --maxlen=200 --dropout_rate=0.2
```

## Misc

The implemention of self attention is modified based on *[this](https://github.com/Kyubyong/transformer)*.

The convergence curve on `MOOCCube`, compared with RNN based approaches:  

<img src="curve.png" width="400">
