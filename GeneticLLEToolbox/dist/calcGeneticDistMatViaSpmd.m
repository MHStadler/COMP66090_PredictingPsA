function [ distMat, t ] = calcGeneticDistMatViaSpmd( X, parPoolSize )
% calcGeneticDistMatViaSpmd Calculate the genetic dist mat in parallel
%
%   [ distMat, t ] = calcGeneticDistMatViaSpmd( X, parPoolSize )
%   
%   Calculate the pairwise genetic distance matrix for X, in parallel,
%   using SPMD. X is a NxD data matrix of additive encoded genetic data, where
%   the rows correspond to samples, and the columns represent features.
%   parPoolSize is the number of cores to use for the distance calculation.
%   distMat is the NxN symmetric matrix with 0s on the diagnoal, where the
%   elements correspond to the pairwise distance between points in X.

[D] = size(X, 2);

minAlleleCounts = sum(X);
majAlleleCounts = sum(2 - X);

minorAlleleNormalization = log10(minAlleleCounts ./ (2 * D));
majorAlleleNormalization = log10(majAlleleCounts ./ (2 * D));

pool = parpool(parPoolSize);

tic;

X_dist = distributed(X);
minorAlleleNormalization_dist = distributed(minorAlleleNormalization);
majorAlleleNormalization_dist = distributed(majorAlleleNormalization);

spmd
    X_local = getLocalPart(X_dist);   
    minorAlleleNormalization_local = getLocalPart(minorAlleleNormalization_dist);
    majorAlleleNormalization_local = getLocalPart(majorAlleleNormalization_dist);
    
    S = vectorizedGeneticSimilarity(X_local, minorAlleleNormalization_local, majorAlleleNormalization_local);    
end
            
similarity = cell2mat(S{1});
for i = 2 : parPoolSize
    similarity = similarity + cell2mat(S{i});
end
similarity = 1 - (similarity ./ (-2 * D));
distMat = squareform(similarity);
t = toc;

delete(pool);

end