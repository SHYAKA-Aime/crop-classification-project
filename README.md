# Crop Classification Using Fused Optical and Radar Remote Sensing Data

This repository contains the complete machine learning and deep learning benchmarking pipeline for classifying agricultural crop types using the Winnipeg Crop Mapping dataset from the UCI Machine Learning Repository. The project investigates whether deep neural networks offer meaningful performance advantages over optimized tree ensembles on high-dimensional multi-sensor agricultural tables.

## Project Structure

- crop_classification_project.ipynb: Main documented notebook containing preprocessing, model training, and evaluations
- requirements.txt: Python package dependencies
- .gitignore: Configuration ensuring data files and models are not committed to GitHub
- README.md: Overview and reproduction instructions for the project

## Dataset Summary

The dataset comes from the UCI Machine Learning Repository and contains 325,834 raw spatial observations across 174 continuous numerical features. The features represent multi-temporal optical spectral bands combined with polarimetric synthetic aperture radar data.

The classification target includes seven crop types:

1. Corn
2. Peas
3. Canola
4. Soybeans
5. Oats
6. Wheat
7. Broadleaf crops

To comply with project instructions, raw data files are not committed to this repository. The complete compressed archive is stored in the associated Google Drive folder.

## Technical Workflow

The notebook is divided into six sequential sections:

1. Data loading, label transformation, stratified 80/20 train-test splitting, and standard feature scaling.
2. Classical machine learning experiments covering linear baselines, random forests, and gradient-boosted decision trees.
3. Input streaming pipelines using the TensorFlow tf.data API with shuffling, batching, and prefetching.
4. Deep learning experiments using the Keras Functional API, testing combinations of Batch Normalization, Dropout, L2 weight decay, and learning rate adjustments.
5. Model diagnostics covering loss trajectories, confusion matrix analysis, class-wise precision and recall, multi-class ROC-AUC, and feature importance.
6. Model weight export to joblib and native Keras formats.

## Experimental Results

Nine distinct experiments were logged and evaluated on the same 52,134 test instances:

| Exp | Approach               | Key Settings                                      | Test Accuracy | Observations                                                           |
| :-- | :--------------------- | :------------------------------------------------ | :------------ | :--------------------------------------------------------------------- |
| 1   | Logistic Regression    | max_iter=1000, solver='lbfgs'                     | 0.9908        | High linear baseline indicates strong feature separability.            |
| 2   | Random Forest          | n_estimators=50                                   | 0.9959        | Non-linear tree partitions reduce classification error.                |
| 3   | XGBoost Default        | tree_method='hist'                                | 0.9975        | Fast histogram binning captures complex sensor interactions.           |
| 4   | XGBoost Optimized      | max_depth=8, lr=0.05, n_estimators=200            | 0.9971        | Yields smooth decision boundaries and a 1.0000 ROC-AUC score.          |
| 5   | LightGBM               | n_estimators=150                                  | 0.9830        | Fast execution, but leaf-wise tree growth performed slightly lower.    |
| 6   | DL Baseline            | Dense(128, 64), Adam                              | 0.9974        | Strong unconstrained baseline without explicit regularization.         |
| 7   | DL Regularized         | Dense(256, 128), BatchNorm, Dropout(0.3, 0.2)     | 0.9981        | Best overall accuracy; prevents node co-adaptation.                    |
| 8   | DL L2 Decay            | Dense(256, 128), L2(0.001)                        | 0.9914        | Weight penalty constrains network capacity, lowering accuracy.         |
| 9   | DL Deeper Architecture | Dense(256, 128, 64), Dropout(0.4, 0.3), lr=0.0005 | 0.9914        | Additional depth did not improve over the two-layer regularized model. |

## Key Findings

1. Deep learning achieved a peak accuracy of 0.9981 in Experiment 7, while the optimized XGBoost model reached 0.9971. For dense tabular data of this nature, gradient boosting achieves comparable predictive accuracy with lower pipeline complexity.
2. The learning curves in Experiment 9 show validation loss tracking lower than training loss during initial epochs. This behavior occurs because Dropout is active during training to penalize node dependence, but disabled during validation.
3. A severe class imbalance exists between dominant classes (such as wheat with over 16,000 samples) and the minority broadleaf class (230 samples). Broadleaf crops recorded a lower F1-score of 0.97 compared to 1.00 across dominant classes.
4. Feature importance extraction identified feature f114 as the strongest individual predictor, followed by f95 and f138, reflecting the physical importance of combined optical reflection and radar backscatter signals.

## Running the Code

Install dependencies:
pip install -r requirements.txt

Run the notebook in Google Colab or Jupyter Notebook:

1. Ensure the raw dataset archive is uploaded to Google Drive under the path specified in cell 1.
2. Execute the notebook sequentially from cell 1 through cell 6.
3. Exported model files will be saved locally and mirrored to Google Drive.
