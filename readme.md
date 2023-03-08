Computer vision for specimen mammography
=============================================

This repository contains code to develop computer vision models that predict margin status of partial mastectomy specimens using intra-operative specimen mammography.

Supplementary Methods
For comparing RadImageNet vs ImageNet models, the number of fully connected layers, neurons, and dropout were held constant to facilitate comparison. After the highest performing model and pre-training dataset were identified, the number of fully connected layers, neurons, dropout, and learning rate were tuned using the training/validation sets. For all comparisons, transfer learning was completed in two phases. First, the base architecture was frozen and the model was trained at a higher learning rate. Second, the base architecture was unfrozen and trained at a lower learning rate. Early stopping was used to avoid overfitting.


1. Comparison of RadImageNet and ImageNet weights across 4 model architectures: search_rin.py
2. Hyperparameter tuning of fully connected layers, number of neurons, dropout, learning rate, image size, and batch: search_rin.py
3. Plotting ROC and PR curves: curves.ipynb
4. Subset analysis of breast density and race/ethnicity: subset.ipynb
5. GradCAM analysis of model attention: heatmap.ipynb
