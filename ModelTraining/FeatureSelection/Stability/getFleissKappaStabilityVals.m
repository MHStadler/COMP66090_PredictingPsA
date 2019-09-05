function [ stability, lower, upper ] = getFleissKappaStabilityVals( featureSets, D )

alpha = 0.05;

M = size(featureSets, 1);

Z = zeros(M, D);
for m = 1:M
    featureSet = featureSets{m, 1};
    
    Z(m, featureSet) = 1;
end

[stability, lower, upper] = getStabilityConfidenceIntervals(Z, alpha);


end

