function [upsampledX, upsampledY] = smote( X, y, minorityClass, upsamplingDegree )
%smote Apply SMOTE to upsample the given dataset
%   
%   [upsampledX, upsampledY] = smote( X, y, minorityClass, upsamplingDegree )
%
%   Uses the SMOTE algorithm as introduced in [1] to generate syntetic
%   samples of the given minorityClass. The upsamplingDegree parameter
%   dictates how many syntetic samples are generated, and can either be a
%   percentage value or the default value 'balanced'. For 'balanced', the
%   algorithm calculates the exact number of new samples that need to be
%   generated to ensure an equal number of minority and majority samples in
%   the final dataset.
%
%   X ... NxD datamatrix, with N samples of D features
%   y ... Nx1 class label vector
%   minorityClass ... class that is considered the minority
%   upsamplingDegree ... optional input parameter that dictates how many
%   syntetic samples are created
%
%   upsampledX ... datamatrix that contains the original data as well as
%   created smote samples
%   upsampledY ... class label vector that contains additional entries for
%   created smote samples
%
%   [1] Chawla, Nitesh V., et al. "SMOTE: synthetic minority over-sampling technique." Journal of artificial intelligence research 16 (2002): 321-357.


if nargin < 4
    upsamplingDegree = 'balanced';
end

majoritySamples = X(y ~= minorityClass, :);
minoritySamples = X(y == minorityClass, :);

[ synteticSamples ] = applySmote(majoritySamples, minoritySamples, upsamplingDegree);

upsampledX = [X;synteticSamples];
upsampledY = [y; zeros(size(synteticSamples, 1), 1) + minorityClass];

end