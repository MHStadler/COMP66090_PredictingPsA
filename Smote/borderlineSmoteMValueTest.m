function [ results ] = borderlineSmoteMValueTest( I, X, y, minorityClass, maxMValue )
%borderlineSmoteMValueTest
%   
%   [ results ] = borderlineSmoteMValueTest( I, X, y, minorityClass, maxMValue )
%
%   Calculate the ratio of minority samples that would be considered
%   endangered in borderlineSmote, by varying the used m from 1 to
%   maxMValue
%
%   I ... number of boostraps
%   X ... NxD data matrix of N samples and D features
%   y ... Nx1 class label vector
%   minorityClass ... class that is considered the minority
%   maxMValue ... the maximum m parameter to test
%
%   results ... IxMaxMValue result matrix, where rows correspond to results
%   of a given bootstrap and columns contain the ratio of endangered minority
%   samples for each tested m parameter

N = size(X, 1);

results = zeros(I, maxMValue);
parfor i = 1:I
    bootInd = randsample(N, N, true);
    
    bootstrapX = X(bootInd, :);
    bootstrapY = y(bootInd);
    
    minoritySamples = X(y == minorityClass, :);
    
    bootstrapValues = zeros(1, maxMValue);
    for m = 1:maxMValue
        endangeredSamples = findEndangeredSamples(bootstrapX, bootstrapY, minorityClass, m);
        
        bootstrapValues(m) = size(endangeredSamples, 1);
    end
    
    results(i, :) = bootstrapValues ./ size(minoritySamples, 1);
end

end