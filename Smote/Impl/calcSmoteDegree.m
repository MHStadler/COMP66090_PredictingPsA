function [upsamplesPerSmoteSample, noExtraSynteticSamples, T] = calcSmoteDegree(noMajoritySamples, noMinoritySamples, noSmoteSamples, upsamplingDegree, smoteMajoritySamples)
    if strcmp('balanced', upsamplingDegree)
        upsamplingDegree = calculateBalancedSamplingDegree(noMajoritySamples, noMinoritySamples, noSmoteSamples, smoteMajoritySamples);
    end
    
    upsamplesPerSmoteSample = floor(upsamplingDegree / 100);
    % If samplingDegree is not an even 100, an additional smoteSample needs to be
    % generated for some samples
    modUpamplingDegree = mod(upsamplingDegree, 100);
    noExtraSynteticSamples = ceil((modUpamplingDegree / 100) * noSmoteSamples);

    if upsamplesPerSmoteSample == 0 
        T = noExtraSynteticSamples;
        upsamplesPerSmoteSample = 1;
        noExtraSynteticSamples = 0;
    else
        T = noSmoteSamples;
    end
end

function [ balanceSamplingDegree ] = calculateBalancedSamplingDegree( noMajoritySamples, noMinoritySamples, noEndangeredSamples, smoteMajoritySamples )

balanceSamplingDegree = 100 * ((noMajoritySamples - noMinoritySamples) / noEndangeredSamples);

% If we also create syntetic samples based on the majority samples, we only
% want do to half the upsampling
if smoteMajoritySamples
    balanceSamplingDegree = balanceSamplingDegree / 2;
end

balanceSamplingDegree = floor(balanceSamplingDegree);

end