# COMP66090 PredictingPsA:

Code developed for MSc Project: Predicting the development of Psoriatic Arthritis using Feature Selection and Locally Linear Embedding

The GeneticLLEToolbox directory contains code to: apply LLE, create out-of-sample extension for new samples, addaptive select the neighbourhood, and code to efficiently calculate the genetic distance in parallel.
LLE code developed based on Matlab Toolbox for Dimensionality Reduction (https://lvdmaaten.github.io/drtoolbox/) developed by Laurens van der Maaten [1].

The Smote directoy contains our implementation of the SMOTE [2] and BorderlineSMOTE [3] algorithms

Lastly, the ModelTraining directoy contains our code for feature selection and for training our models in cross validation & external validation, as well as the code for evaluation.

[1] Levina, Elizaveta, and Peter J. Bickel. "Maximum likelihood estimation of intrinsic dimension." Advances in neural information processing systems. 2005.
[2] Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357.
[3] Han, Hui, Wen-Yuan Wang, and Bing-Huan Mao. "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning." International conference on intelligent computing. Springer, Berlin, Heidelberg, 2005.
