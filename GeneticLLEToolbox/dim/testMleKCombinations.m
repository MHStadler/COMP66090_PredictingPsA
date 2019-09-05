function [ mValues ] = testMleKCombinations( X )
% testMleKCombinations Test the MLE estimator for X with a number of k
% combinations
%
%   [ mValues ] = testMleKCombinations( X )
%
%   Estimate the intrinsic dimensionality of X for a range of kValue
%   combinations, using the MLE estimator. The output mValues contains the
%   estimated dimensionalities for each combination

nn = 1;
for i = 5:5:40
    for j = i:5:30
       kValues(nn, 1) = i;
       kValues(nn, 2) = j+i;
       
       nn = nn + 1;
    end
end

mValues = estimateIntrinsicDimensionalityViaMle(X, kValues);

end