# Online Label Shift



## Setup Environment

```
conda env update --file environment.yml
```


```
pip install --upgrade "jax[cpu]"
```
```
pip install optax
```
## Train model

This script supports model training for synthetic, cifar10, and mnist datasets.

```
python scripts/train_model.py -d <data> -m <model> -e <epoch>
```

The corresponding models are:

| Data      | Model    |
| --------- | -------- |
| mnist     | MLP      |
| cifar10   | resnet18 |



## Run Online Label Shift Experiment

The following command runs the online label shift experiment.
It expects the base model has been trained and saved under `/model`

```
python scripts/run_ols.py -d synthetic -m logreg --do-all 1 -t 1000 --save 1
```

To see all the options

```
python scripts/run_ols.py -h
```

