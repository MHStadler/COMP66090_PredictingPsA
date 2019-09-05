function [ Y ] = locallyLinearEmbedding( X, d, distMat, kValues )
% locallyLinearEmbedding Calculate embedding via LLE
%
%   [ Y ] = locallyLinearEmbedding( X, d, distMat, kValues )
%
%   Calculate low dimensional embedding via LLE. X is a NxD data matrix,
%   with N samples, and D features. d is the target dimensionality of the
%   embedding, and distMat is the precalculated distance used to determine
%   the neihgbours for each sample in X. kValues is a Nx1 vector that
%   contains the number of neighbours to use for each point in X.
%   The output Y is a Nxd matrix, containing the low dimensional embedding
%   for X. Implements the LLE algorithm as introduced by [1].
%
% Based on the Matlab Toolbox for Dimensionality Reduction developed by
% Laurens Van Der Maaten[2]
%
% [1] Saul, Lawrence K., and Sam T. Roweis. "Think globally, fit locally: unsupervised learning of low dimensional manifolds." Journal of machine learning research 4.Jun (2003): 119-155.
% [2] Van Der Maaten, Laurens, Eric Postma, and Jaap Van den Herik. "Dimensionality reduction: a comparative." J Mach Learn Res 10.66-71 (2009): 13.

X = X';
[D, N] = size(X);

[~, Idx] = sort(distMat, 2);

if(max(kValues) > D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(N, N);
for i = 1:N
    k = kValues(i);
    neighbourhood = Idx(i, 2:k+1);
    
    Z = X(:, neighbourhood);
    Xi = X(:, i);
    
    Z = Z - repmat(Xi, [1, k]);
    
    C = Z' * Z;
    
    t = trace(C);
    if(t > 0) 
        C = C + eye(k,k)*tol*trace(C); 
    else
        C = C + eye(k,k)*tol;
    end
    
    w = C\ones(k,1);
    
    W(i, neighbourhood) = w;
end

W = W ./ sum(W, 2);

I = eye(N, N);

M = (I-W);
M = M' * M;
M(isnan(M)) = 0;
M(isinf(M)) = 0;

options.disp = 0;
options.isreal = 1;
options.issym = 1;
options.tol = 1e-6;

[mappedX, eigenvals] = eigs(M + eps * eye(N), d + 1, 0, options);          % only need bottom (no_dims + 1) eigenvectors

[~, ind] = sort(diag(eigenvals), 'ascend');
if size(mappedX, 2) < d + 1
    no_dims = size(mappedX, 2) - 1;
    warning(['Target dimensionality reduced to ' num2str(no_dims) '...']);
end

Y = mappedX(:,ind(2:d + 1));                                % throw away zero eigenvector/value

end