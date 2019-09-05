function [ externalValidationResults ] = trainModelsAndRunExternalValidation( X, y, modelName, featureSelectionOutput, cvResults, cvParams, valX, valY )

tic;

D = size(cvResults, 2);

fsMethods = fieldnames(featureSelectionOutput);
noModels = numel(fsMethods);
noParams = D / noModels;

paramIdxs = getParamIdxs(cvResults, noModels, noParams);
hyperParameters = getHyperParameters(cvParams, paramIdxs);

[X, y] = borderlineSmote(X, y, 0, 9);

externalValidationResults = cell(1, 2);

classifiers = cell(noModels, 1);
evalResults = cell(noModels, 101);

parfor i = 1:noModels
    featureSet = featureSelectionOutput.(fsMethods{i}).featureSet;
    
    classifier = trainClassifier(X(:, featureSet), y, modelName, hyperParameters{i}, 1);
    evaluation = classifier.classify(valX(:, featureSet), valY);
    
    bootstrapEvalResults = cell(101, 1);
    bootstrapEvalResults{1} = evaluation;
    
    N = size(valX, 1);
    for m = 2:101
        bootInd = randsample(N, N, true);
        
        bootValX = valX(bootInd, featureSet);
        bootValY = valY(bootInd);
        
        bootEval = classifier.classify(bootValX, bootValY);
        bootstrapEvalResults{m} = bootEval;
    end
    
    classifiers{i} = classifier;
    evalResults(i, :) = bootstrapEvalResults;
end

externalValidationResults{1} = classifiers;
externalValidationResults{2} = evalResults;

toc;

end

function [paramIdxs] = getParamIdxs(resultsStruct, noModels, noParams) 
    kappa = [resultsStruct.KAPPA];
    kappa = reshape(kappa, 5, []);

    paramIdxs  = zeros(noModels, 1);
    for n = 1:noModels
        idx = (n-1) * noParams;

        kappaVals = kappa(:, idx+1:idx+noParams);

        [~, maxIdx] = max(mean(kappaVals));

        paramIdxs(n) = maxIdx + idx;
    end
end

function [hyperParameters] = getHyperParameters(cvParams, paramIdxs)

params = reshape(cvParams(:, 3), 5, []);

hyperParameters = params(1, paramIdxs);

end