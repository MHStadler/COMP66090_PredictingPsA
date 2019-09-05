function [ featureSubset ] = linearAggregationEnsembleFeatureSelection( cvFeatures, D )
%linearAggregationEnsembleFeatureSelection Select the featureSubset from an
%ensemble of featureSubsets yielded in CV
%
%   [ featureSubset ] = linearAggregationEnsembleFeatureSelection( cvFeatures, D )
%
% Select the final feature subset from the subsets yielded by
% CrossValidation, using Linear Aggregation Ensemble Feature Selection [1].
%
% The N features selected in CV are ranked, with the top feature being
% ranked as 1, and any n-th feature getting rank n. Any not selected
% feature is ranked as N+1.
%
% The feature ranks across the different folds are then summed up, and the
% N features with the smallest rank are chosen as the featureSubset.
%
% cvFeatures ... Feature Indices selected in each CV fold kxN, where k is
% the number of folds and N is the number of features selected in each fold
% D ... number of features in the dataset
%
% featureSubset ... 1xd Vector containing the Feature Indices of the
% selected featureSubset
%
% [1] Abeel, Thomas, et al. "Robust biomarker identification for cancer diagnosis with ensemble feature selection methods." Bioinformatics 26.3 (2009): 392-398.

[noSets, noFeatures] = size(cvFeatures);

% Init weights with noFeatures + 1
featureMatrix = ones(noSets, D) * noFeatures + 1;

for i = 1:noSets
    featureSubset = cvFeatures(i, :);
    
    noNonZeroElements = nnz(featureSubset(:) ~= 0);
    
    % Rank features 1:N, with top feature being ranked 1
    featureMatrix(i, featureSubset(1:noNonZeroElements)) = 1:noNonZeroElements;
end

% Calculate each features final rank
rankedFeatures = sum(featureMatrix);
[~, minRankedFeatureIndices] = sort(rankedFeatures, 'ascend');

% Select the N bottom ranked features as featureSubset
featureSubset = minRankedFeatureIndices(1:noFeatures);

end

