Computer vision for specimen mammography
=============================================

This repository contains code to develop computer vision models that predict margin status of partial mastectomy specimens using intra-operative specimen mammography.

1. Comparison of RadImageNet and ImageNet weights across 4 model architectures: search_rin.py
2. Hyperparameter tuning of fully connected layers, number of neurons, dropout, learning rate, image size, and batch: search_rin.py
3. Plotting ROC and PR curves: curves.ipynb
4. Subset analysis of breast density and race/ethnicity: subset.ipynb
5. GradCAM analysis of model attention: heatmap.ipynb