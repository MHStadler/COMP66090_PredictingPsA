function [ classifier ] = trainClassifier( X, y, classifierType, hyperParameters, positiveClass )

c = str2func(classifierType);

classifier = c(X, y, hyperParameters, positiveClass);

end