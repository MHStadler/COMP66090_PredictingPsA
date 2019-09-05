# COMP66090 PredictingPsA:

Code developed for MSc Project: Predicting the development of Psoriatic Arthritis using Feature Selection and Locally Linear Embedding

The GeneticLLEToolbox directory contains code to: apply LLE, create out-of-sample extension for new samples, adaptively select the neighbourhood, and code to efficiently calculate the genetic distance in parallel.
LLE code developed based on Matlab Toolbox for Dimensionality Reduction (https://lvdmaaten.github.io/drtoolbox/) developed by Laurens van der Maaten [1].

The Smote directoy contains our implementation of the SMOTE [2] and BorderlineSMOTE [3] algorithms

Lastly, the ModelTraining directoy contains our code for feature selection and for training our models in cross validation & external validation, as well as the code for evaluation.

The libs directory contains third party software:
 - FEAST[4]: http://www.cs.man.ac.uk/~gbrown/fstoolbox/
 - Noguera's Fleiss Kappa Stability [5]: https://github.com/nogueirs/JMLR2018

# References:

[1] Levina, Elizaveta, and Peter J. Bickel. "Maximum likelihood estimation of intrinsic dimension." Advances in neural information processing systems. 2005.

[2] Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357.

[3] Han, Hui, Wen-Yuan Wang, and Bing-Huan Mao. "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning." International conference on intelligent computing. Springer, Berlin, Heidelberg, 2005.

[4] Brown, Gavin, et al. "Conditional likelihood maximisation: a unifying framework for information theoretic feature selection." Journal of machine learning research 13.Jan (2012): 27-66.

[5] Nogueira, Sarah, Konstantinos Sechidis, and Gavin Brown. "On the Stability of Feature Selection Algorithms." Journal of Machine Learning Research 18 (2017): 174-1.
