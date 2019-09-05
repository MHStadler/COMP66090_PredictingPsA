function [ featureSet, featureSetStab, B, FitInfo, BStab, FitInfoStab] = elasticNetCVFeatureSelection( X, y, alpha, noFolds, performUpsampling )

noLambdas = 100;

if performUpsampling
    [X, y] = borderlineSmote(X, y, 0, 9);
end

lambdaValues = calcLambda(X, y, alpha, noLambdas);
mseValues = zeros(noFolds, noLambdas);

foldFeatureSets = cell(noFolds, noLambdas);

cv = cvpartition(y, 'kfold', noFolds);
parfor i = 1:noFolds
    train = cv.training(i);
    test = cv.test(i);
    
    trainingData = X(train, :);
    trainingLabels = y(train);
    
    % ... fit the lasso, ...
    [B, FitInfo] = lasso(trainingData, trainingLabels, 'Alpha', alpha, 'Lambda', lambdaValues); 
    
    foldFeatures = selectFeaturesFromLassoCoefficients(B);
    foldFeatureSets(i, :) = foldFeatures;
    
    % ... and calculate the mean MSE for this fit, for each lambda
    yhat = X(test, :) * B + FitInfo.Intercept;
    mse = mean(bsxfun(@minus, y(test), yhat).^2);
    mseValues(i, :) = mse;
end

stability = getStabilityForFoldFeatures(foldFeatureSets, size(X, 2));

% Mean of MSE for each lambda, across the folds, as well as SE for each
% value
meanMse = mean(mseValues);
[minMSE, minIx] = min(meanMse);

% Find lambda with mean MSE closest to minMSE + 1SE
se = std(mseValues(:, minIx)) / sqrt(size(mseValues, 1));
seIdxs = find(meanMse <= (minMSE + se));

seIdx = seIdxs(end);

% Parsimonious Idxs within 1se of minMSE
idxRange = minIx:seIdx;
[~, maxStabIdx] = max(stability(idxRange));
maxStabSeIdx = idxRange(maxStabIdx);

% The Lasso function reverses it output, so we need to reverse lambda here
% to ensure columns of mseValues correctly align with columns in lambda
reversedLambda = lambdaValues(noLambdas:-1:1);
lambda = reversedLambda(seIdx);
lambdaMaxStab = reversedLambda(maxStabSeIdx);

% Perform lasso on whole set to get feature set
[B, FitInfo] = lasso(X, y, 'Alpha', alpha, 'Lambda', lambda);
[featureSets] = selectFeaturesFromLassoCoefficients(B);

[BStab, FitInfoStab] = lasso(X, y, 'Alpha', alpha, 'Lambda', lambdaMaxStab);
[featureSetsStab] = selectFeaturesFromLassoCoefficients(BStab);

featureSet = featureSets{1};

featureSetStab = featureSetsStab{1};

end

function [stabilityValues] = getStabilityForFoldFeatures(foldFeatureSets, D) 

N = size(foldFeatureSets, 2);

stabilityValues = zeros(1, N);
for i = 1:N
    featureSets = foldFeatureSets(:, i);
    
    stabilityValues(i) = getFleissKappaStabilityVals(featureSets, D);
end

end

function [ featureSets ] = selectFeaturesFromLassoCoefficients(B) 
    noBVals = size(B, 2);
    featureSets = cell(1, noBVals);
    
    [~, idx] = sort(abs(B), 'descend');
    
    for i = 1:noBVals
        % Calc number of non zero coefficients in this fold
        noNonZeroCoefficients = nnz(B(:, i) ~= 0);
        
        featureSets{i} = idx(1:noNonZeroCoefficients);
    end
end