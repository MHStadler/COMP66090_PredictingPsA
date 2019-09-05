function [kValues] = findNNAdaptive(X, no_dims, max_k, distMat)
%findNNAdaptive Find adaptive neighbourhood
%
%	[kValues] = findNNAdaptive(X, no_dims, max_k, distMat)
%
%   Find the right number of neighbours for each point, using adaptive
%   neighbourhood selection. The rows in X correspond to samples, and the
%   columns to features. no_dims is the number of dimensions of the target
%   embedding, and max_k is the maximum number of neighbours. distMat is a
%   the pre-calculated distance matrix, if none is provided the Euclidean
%   distance is used.
%   The output kValues contains the a suitable number of neighbours k for
%   every point in X.
%
% Based on the Matlab Toolbox for Dimensionality Reduction developed by
% Laurens Van Der Maaten[1]
%
% [1] Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik. "Dimensionality reduction: a comparative." J Mach Learn Res 10.66-71 (2009): 13.

eucDistMat = squareform(pdist(X));

if nargin < 4
    distMat = eucDistMat;
end

N = size(X, 1);
min_k = no_dims + 1;
    
kValues = zeros(N, 1);

X = X';
    
% For all datapoints
parfor i=1:N
    % current data point
    p = X(:,i);

    % Use distmat to find neighbours
    d = distMat(i, :);
    [~, ind] = sort(d);
    
    % ... but euclidean distance to calc linearity
    d2 = eucDistMat(i, :);
    d2 = d2(ind);
    
    % Estimate local tangent space by updating the number of neighbors k
    stop = 0;
    k = min_k;
    
    while ~stop && k + 1 < max_k
        k = k + 1;
        
        % Estimate local tangent space (for current value of k)
        tmpX = X(:,ind(2:k + 1)) - repmat(p, 1, k);
        lambda = svd(tmpX);
        [lambda, ind2] = sort(lambda, 'descend');
        if length(lambda) < no_dims
            break;
        end
        
        % Estimate T_{1}
        T = (1 / k) ^ (1 / no_dims) * d2(k + 1);
         
        % Check whether stop condition is violated
        if lambda(no_dims) >= T
            stop = true;
        end
    end

    % Compute tangent space at k - 1 since k failed stop condition
    [U, lambda, M] = svd(tmpX(:,1:k-1));
    [lambda, ind2] = sort(diag(lambda), 'descend');
	U = U(:,ind2(1:no_dims))';
    
    % Select neighbors that correspond to the local tangent space
    stop = 0;
    while ~stop && k + 1 < max_k
        k = k + 1;
        
        % Projection of (x_{k} - x_{i}) onto tangent space
        onto = sum( (U * (X(:,ind(k + 1)) - p)).^2 );
        
        % Check whether stop condition is violated
        if d2(k + 1)^2 - onto > T^2
            k = k - 1;
            stop = true;
        end
    end
    
    kValues(i) = k;
end

end