# CropClassification-TCN
 Image Interpretation Lab 3

## Models

1. Simple TCN

2. TCN with Residual Blocks

Both models can be trained either using weighted loss or not. We also implemented random prediction as a simple baseline.

### Train

Please go to `CodeRepo/MODEL_NAME/train_config.py` to change the training configuration. Then run `CodeRepo/MODEL_NAME/train.py`

The code supports `wandb.ai` to monitor the training procedure online.

### Automatic Hyperparameter Tuning

We tuned hyperparameters on simple TCN. Please run `CodeRepo/simple_tcn/hyperpara_opt.py`.

### Test

Please go to `CodeRepo/MODEL_NAME/test_config.py` to change the testing configuration. Then run `CodeRepo/MODEL_NAME/predict.py`. It will save the confusion matrix and metrics (i.e., f1, precision and recall per crop type) as `.csv` file.

#### Generate crop type map and error map

Please go to `CodeRepo/MODEL_NAME/test_config.py` to change the testing configuration. Then run `CodeRepo/MODEL_NAME/predict_map.py`.

### Evaluations  

#### Visualize crop type map and error map

Please go to `CodeRepo/utils/visualize_tf_pred.py`

#### Result evaluation and visualization

The analysis can be found in `ResultEvaluation.ipynb`


