function [ dist ] = pairwiseGeneticDistance( p, X, minorAlleleNormalization, majorAlleleNormalization )
%pairwiseGeneticDistance Calculate the genetic distance between p and samples in X
%
%   pairwiseGeneticDistance( p, X, minorAlleleNormalization, majorAlleleNormalization )
%
%   Calculate the pairwise genetic distance between the point p and points
%   in X. p is a 1xD vector of new points, and X is the NxD matrix of
%   original datapoints. minorAlleleNormalization, and 
%   majorAlleleNormalization are the normalization applied to the shared
%   minor and major alleles respectively. The output dist is a 1xN vector
%   containing the distance between the point p and all points in X.

D = size(X, 2);

% Calc minor alleles shared as the min between the 2 minor allele
% counts and weight it using the minor allele frequency         
minorAllelesShared = (min(p, X) .* minorAlleleNormalization');
    
% Calc major alleles shared as 2 - max of the 2 minor allele
% counts, and weight it using the major allele frequency
majorAllelesShared = ((2 - max(p, X)) .* majorAlleleNormalization');

% Calc similarity
s = sum(minorAllelesShared + majorAllelesShared, 2);

dist = 1 - (s ./ (-2 * D));

end

