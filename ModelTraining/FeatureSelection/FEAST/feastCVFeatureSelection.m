function [ featureSet, cvFeatures ] = feastCVFeatureSelection( X, y, technique, noFeatures, noFolds )

cvFeatures = zeros(noFolds, noFeatures);
cv = cvpartition(y, 'kfold', noFolds);
parfor i = 1:noFolds
    fold = cv.training(i);
    
    foldData = X(fold, :);
    foldY = y(fold);
    
    [ featureSubset ] = feast(technique, noFeatures, discretize(foldData, 8), foldY);
    cvFeatures(i, :) = featureSubset;
end

featureSet = linearAggregationEnsembleFeatureSelection(cvFeatures, size(X, 2));

end