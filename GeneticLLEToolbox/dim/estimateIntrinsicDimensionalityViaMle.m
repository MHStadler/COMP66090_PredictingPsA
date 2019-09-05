function [ mValues ] = estimateIntrinsicDimensionalityViaMle( X, kVals)
%estimateIntrinsicDimensionalityViaMle Estimate the intrinsic
%dimensionality of X, using MLE
%
%   [ mValues ] = estimateIntrinsicDimensionalityViaMle( X, kVals)
%
%   Estimate the intrisinc dimensionality of X using a MLE estimator. X is
%   a NxD data matrix of N samples, with D features. kVals is a Mx2 vector
%   of k value combinations. mValues returns the estimated dimensionality 
%   for each kValue combination 
%
% [1] Levina, Elizaveta, and Peter J. Bickel. "Maximum likelihood estimation of intrinsic dimension." Advances in neural information processing systems. 2005.

N = size(X, 1);  

X = X - repmat(mean(X, 1), [N 1]);
X = X ./ repmat(var(X, 1) + 1e-7, [N 1]);

D = sort(squareform(pdist(X)), 2);   

nK = size(kVals, 1);
mValues = zeros(nK, 1);
for n = 1:nK
    k1 = kVals(n, 1);
    k2 = kVals(n, 2);
    
    mkValues = zeros(N, 1);
    parfor i = 1:N
        lD = D(i, :);
        kNeighbours = lD(1, 2:k2+1);

        for k = k2 : -1 : k1
            m_k = 1 / (sum(log(kNeighbours(k) ./ kNeighbours(1:k-1))) / (k-1));

            mkValues(i) = mkValues(i) + m_k;
        end
    end
    
    m = (sum(mkValues) / N) / (k2 - k1 + 1);
    
    mValues(n) = m;
end

end