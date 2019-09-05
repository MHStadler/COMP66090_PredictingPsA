function [ outOfSamplePoints ] = createLleOutOfSamplePoints( points, X, Y, max_k, is_genetic )
%createLleOutOfSamplePoints Create out-of-sample embeddings for new points
%
%   [ outOfSamplePoints ] = createLleOutOfSamplePoints( points, X, Y, no_dims, max_k, is_genetic )
%
%   Calculate the out-of-sample embedding for new points. X is the original
%   data, and Y is their embedding. Neighbours for points in X are found
%   using adaptive neighbourhood selection, and max_k represents the
%   maximum neighbourhood size. is_genetic is a boolean flag that indicates
%   whether the genetic distance should be used to find neighbours. By
%   default, or if fales, the Euclidean distance is used instead.
%   The output outOfSamplePoints contains the embedding for the new points.
%
%   Implements extension algorithm introduced by [1].
%
% Based on the Matlab Toolbox for Dimensionality Reduction developed by
% Laurens Van Der Maaten[2]
%
% [1] Saul, Lawrence K., and Sam T. Roweis. "Think globally, fit locally: unsupervised learning of low dimensional manifolds." Journal of machine learning research 4.Jun (2003): 119-155.
% [2] Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik. "Dimensionality reduction: a comparative." J Mach Learn Res 10.66-71 (2009): 13.

no_dims = size(Y, 2);

pN = size(points, 1);
outOfSamplePoints = zeros(pN, no_dims);

minAlleleCounts = sum(X);
majAlleleCounts = sum(2 - X);

minorAlleleNormalization = log10(minAlleleCounts ./ (2 * D));
majorAlleleNormalization = log10(majAlleleCounts ./ (2 * D));

parfor pn = 1:pN
    p = points(pn, :);
    
    if is_genetic
        d = pairwiseGeneticDistance(p, X, minorAlleleNormalization, majorAlleleNormalization);
    else 
        d = pdist2(p, X);
    end
    [~, ind] = sort(d);
    
    adaptiveNN = findAdaptiveNNForPoint(p, X, no_dims, max_k, ind);
    
    outOfSamplePoint = lleOutOfSampleExtension(p, X, Y, numel(adaptiveNN), adaptiveNN);
    
    outOfSamplePoints(pn, :) = outOfSamplePoint;
end

end

function [adaptiveNN] = findAdaptiveNNForPoint(p, X, no_dims, max_k, ind)
    d2 = pdist2(p, X);
    d2 = d2(ind);
    
    X = X';
    p = p';
    
    % Estimate local tangent space by updating the number of neighbors k
    stop = 0;
    k = no_dims + 1;
    
    while ~stop && k + 1 < max_k
        % Update k
        k = k + 1;
        
        % Estimate local tangent space (for current value of k)
        tmpX = X(:,ind(1:k)) - repmat(p, 1, k);
        lambda = svd(tmpX);
        [lambda, ~] = sort(lambda, 'descend');
        if length(lambda) < no_dims
            break;
        end
        
        % Estimate T_{1}
        T = (1 / k) ^ (1 / no_dims) * d2(k);
         
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
        % Update k
        k = k + 1;
        
        % Projection of (x_{k} - x_{i}) onto tangent space
        onto = sum( (U * (X(:,ind(k)) - p)).^2 );
        
        % Check whether stop condition is violated
        if d2(k)^2 - onto > T^2
            k = k - 1;
            stop = true;
        end
    end
    
    adaptiveNN = ind(1:k);
end