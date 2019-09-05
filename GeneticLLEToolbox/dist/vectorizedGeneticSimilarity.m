function [ S ] = vectorizedGeneticSimilarity( X, minorAlleleNormalization, majorAlleleNormalization )
%vectorizedGeneticSimilarity Calculate the genetic similarity for data in X
%
%   [ S ] = vectorizedGeneticSimilarity( X, minorAlleleNormalization, majorAlleleNormalization )
%
%   Calculate the pairwise genetic similarity for NxD data matrix X of 
%   N samples with D additive encoded SNP features, where each feature
%   denotes the number of minor alleles at a given SNP.
%   Returns a N-1x1 cell array S, where the nth row is a vector containing the
%   similarities between X(n, :) and the elements X(n+1:N, :).

N = size(X, 1);
S = cell(N-1, 1);

for i = 1:N-1
    x = X(i, :);
    y = X(i+1:end, :);
   
    % Calc minor alleles shared as the min between the 2 minor allele
    % counts and weight it using the minor allele frequency         
    minorAllelesShared = (min(x, y) .* minorAlleleNormalization);
    
    % Calc major alleles shared as 2 - max of the 2 minor allele
    % counts. Weight it using the major allele frequency
    majorAllelesShared = ((2 - max(x, y)) .* majorAlleleNormalization);
    
    % Calc similarity
    s = sum(minorAllelesShared + majorAllelesShared, 2);
    
    S{i} = s;
end

end