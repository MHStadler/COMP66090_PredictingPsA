function [ upsampledX, upsampledY] = borderlineSmote( X, y, minorityClass, m, smoteMajoritySamples, upsamplingDegree)
%smote Apply Borderline SMOTE to upsample the given dataset
%   
%   [upsampledX, upsampledY] = smote( X, y, minorityClass, upsamplingDegree )
%
%   Uses the Borderline SMOTE algorithm as introduced in [1] to generate syntetic
%   samples of the given minorityClass. The upsamplingDegree parameter
%   dictates how many syntetic samples are generated, and can either be a
%   percentage value or the default value 'balanced'. For 'balanced', the
%   algorithm calculates the exact number of new samples that need to be
%   generated to ensure an equal number of minority and majority samples in
%   the final dataset.
%   If smoteMajoritySamples is true, Borderline2 SMOTE is used instead,
%   which creates some additional samples that lie between the minority
%   sample and its majority class neighbours.
%
%   X ... NxD datamatrix, with N samples of D features
%   y ... Nx1 class label vector
%   minorityClass ... class that is considered the minority
%   m ... neighbourhood size used to determine which samples are considered
%   endangered
%   smoteMajoritySamples ... optional boolean flag that dictates whether
%   syntetic samples close to majority samples should be created as well
%   upsamplingDegree ... optional input parameter that dictates how many
%   syntetic samples are created
%
%   upsampledX ... datamatrix that contains the original data as well as
%   created smote samples
%   upsampledY ... class label vector that contains additional entries for
%   created smote samples
%
%   [1] Han, Hui, Wen-Yuan Wang, and Bing-Huan Mao. "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning." International conference on intelligent computing. Springer, Berlin, Heidelberg, 2005.

if nargin < 5
    smoteMajoritySamples = false;
end

if nargin < 6
    upsamplingDegree = 'balanced';
end

majoritySamples = X(y ~= minorityClass, :);
minoritySamples = X(y == minorityClass, :);

endangeredSamples = findEndangeredSamples(X, y, minorityClass, m);

[ synteticSamples ] = applySmote(majoritySamples, minoritySamples, upsamplingDegree, endangeredSamples, smoteMajoritySamples);

upsampledX = [X;synteticSamples];
upsampledY = [y; zeros(size(synteticSamples, 1), 1) + minorityClass];

end