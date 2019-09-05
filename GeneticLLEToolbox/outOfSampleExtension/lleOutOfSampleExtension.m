function point = lleOutOfSampleExtension( p, X, Y, k, ind )
%lleOutOfSampleExtension Calculate the out-of-sample extension for p
%
%   point = lleOutOfSampleExtension( p, X, Y, k, ind )
%
%   Caclculate the out-of-sample extension for a new point p. X is the
%   original data, and Y is the derived embedding. k is the number of
%   neighbours to use, and ind is a sorted vector that contains the indices
%   for the nearest neighbours of p in X.
%
% Based on the Matlab Toolbox for Dimensionality Reduction developed by
% Laurens Van Der Maaten[1]
%
% [1] Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik. "Dimensionality reduction: a comparative." J Mach Learn Res 10.66-71 (2009): 13.

N = size(X, 1);
d = size(Y, 2);

if nargin < 5
    D = pdist2(p, X);
    [~, ind] = sort(D, 2, 'ascend');
end

% Calculate Gram Matrix kxd, where the kth row is the point - its kth
% nearest neighbour
C = repmat(p, [k 1]) - X(ind(1:k), :);
C = C * C';

% Calc weights for Gram Matrix
invC = inv(C);
W = sum(invC, 2) ./ sum(sum(invC));

% Init Kernel Matrix with weights
K = zeros(N, 1);
K(ind(1:k)) = W;

% Point as Embedding .* weights
point = sum(Y .* repmat(K, [1 d]), 1);

end