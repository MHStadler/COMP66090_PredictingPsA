function [ balanceSamplingDegree ] = calculateBalancedSamplingDegree( noMajoritySamples, noMinoritySamples, noEndangeredSamples, smoteMajoritySamples )

if nargin < 3
    noEndangeredSamples = noMinoritySamples;
end

if nargin < 4
    smoteMajoritySamples = false;
end

balanceSamplingDegree = floor(100 * ((noMajoritySamples - noMinoritySamples) / noEndangeredSamples));

if smoteMajoritySamples
    balanceSamplingDegree = ceil(balanceSamplingDegree / 2);
end

end