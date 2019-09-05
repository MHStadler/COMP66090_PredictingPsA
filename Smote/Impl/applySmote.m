function [ synteticSamples ] = applySmote(majoritySamples, minoritySamples, upsamplingDegree, smoteSamples, smoteMajoritySamples )

if nargin < 5
    smoteSamples = minoritySamples;
end

if nargin < 6
    smoteMajoritySamples = false;
end

K = 5;
D = size(minoritySamples, 2);

noMajoritySamples = size(majoritySamples, 1);
noMinoritySamples = size(minoritySamples, 1);
noSmoteSamples = size(smoteSamples, 1);

[upsamplesPerSmoteSample, noExtraSynteticSamples, T] = calcSmoteDegree(noMajoritySamples, noMinoritySamples, noSmoteSamples, upsamplingDegree, smoteMajoritySamples);

noSynteticSamples = T * upsamplesPerSmoteSample + noExtraSynteticSamples;

% Find K+1 minority sample neighbours
idx = knnsearch(minoritySamples, smoteSamples(1:T, :), 'K', K + 1);
synteticSamples = zeros(noSynteticSamples, D);

if smoteMajoritySamples
    majIdx = knnsearch(majoritySamples, smoteSamples(1:T, :), 'K', K + 1);
    majSynteticSamples = zeros(noSynteticSamples, D);
end

synteticSamplesIdx = 0;
for i = 1:T
    noSamples = upsamplesPerSmoteSample;
    if i <= noExtraSynteticSamples
        noSamples = noSamples + 1;
    end
    startIdx = synteticSamplesIdx + 1;
    endIdx = synteticSamplesIdx + noSamples;
    
    smoteSample = smoteSamples(i, :);
    
    neighbours = minoritySamples(idx(i, 2:end), :);
    [ newSamples ] = createSmoteSamples(smoteSample, neighbours, noSamples);
    synteticSamples(startIdx:endIdx, :) = newSamples;
    
    if smoteMajoritySamples
        majNeighbours = majoritySamples(majIdx(i, 2:end), :);
        [ newSamples ] = createSmoteSamples(smoteSample, majNeighbours, noSamples, 0.5);
        majSynteticSamples(startIdx:endIdx, :) = newSamples;
    end
    
    synteticSamplesIdx = synteticSamplesIdx + noSamples;
end

if smoteMajoritySamples
    synteticSamples = [ synteticSamples; majSynteticSamples];
end

end