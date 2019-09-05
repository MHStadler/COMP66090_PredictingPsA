function [ smoteSamples ] = createSmoteSamples(minoritySample, neighbours, noSmoteSamples, maxWeight)
    if nargin < 4
        maxWeight = 1;
    end

    D = size(minoritySample, 2);
    K = size(neighbours, 1);
    
    % Choose random neighbours to use for upsampling
    randomNeighboursIdx = randi([1, K], 1, noSmoteSamples);
    randomNeighbours = neighbours(randomNeighboursIdx, :);
    
    delta = randomNeighbours - minoritySample;
    
    randomFeatureWeights = maxWeight * rand([noSmoteSamples, D]);
    smoteSamples = minoritySample + (delta .* randomFeatureWeights);
end