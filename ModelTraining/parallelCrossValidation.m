function [ cvResults, foldParams ] = parallelCrossValidation( X,  y, config, modelName, featureSelectionOutput )

tic;

config = config.ModelTraining;
noOuterFolds = config.noOuterFolds;

[model, hyperParameters] = getModelAndParams(config, modelName);

folds = getFolds(y, noOuterFolds);
foldParams = buildParLoopParams(folds, hyperParameters, featureSelectionOutput);

cvResults = trainAndEvaluateModels(X, y, model, foldParams);
cvResults = reshape(cvResults, noOuterFolds, []);
cvResults = getResultsStruct(cvResults);

toc;

end

function [model, hyperParameters] = getModelAndParams(config, name)
    noModels = numel(config.models);
    
    for i = 1:noModels
        model = config.models{i};
        
        if(strcmp(name, model.name))
            hyperParameters = buildHyperParameterArray(model.hyperParameters);
            break;
        end
    end
end

function [folds] = getFolds(y, noFolds)
    folds = cell(noFolds, 2);

    cvFolds = cvpartition(y, 'kfold', noFolds);
    
    for i = 1:noFolds
        folds{i, 1} = cvFolds.training(i);
        folds{i, 2} = cvFolds.test(i);
    end
end

function [ hyperParametersArray ] = buildHyperParameterArray(hyperParameters) 
    % Extract all fields of struct
    fields = fieldnames(hyperParameters);
    vec = cell(numel(fields), 1);
    for i=1:numel(fields)
        vec{i} = hyperParameters.(fields{i});
    end

    dim = length(vec):-1:1 ;

    % create 
    [A{dim}] = ndgrid(vec{dim});

    % Generate all combination matrix
    mat = reshape(cat(dim(1)+1,A{:}),[],dim(1));

    % convert all combination matrix to table
    tbl = cell2table(num2cell(mat),'VariableName',fields);

    hyperParametersArray = table2struct(tbl);
end

function [fullParams] = buildParLoopParams(innerFolds, hyperParameters, featureSelectionOutput)
    noFolds = size(innerFolds, 1);
    noParamCombs = size(hyperParameters, 1);
    
    featureSetNames = fieldnames(featureSelectionOutput);
    noFeatureSets = numel(featureSetNames);
    
    fullParams = cell(noFolds * noParamCombs * noFeatureSets, 4);
    
    fullParams(:, 1:2) = repmat(innerFolds, noParamCombs * noFeatureSets, 1);
    
    % Group HyperParams for each Fold
    hyperParams = cell(noFolds * noParamCombs, 1);
    for j = 1:noParamCombs
        parameter = cell(1, 1);
        parameter{1} = hyperParameters(j);
        parameters = repmat(parameter, noFolds, 1);
    
        idx = (j-1) * noFolds;
    
        hyperParams(idx+1:idx+noFolds, 1) = parameters;
    end
    fullParams(:, 3) = repmat(hyperParams, noFeatureSets, 1);
    
    % Set Feature Sets
    for j = 1:noFeatureSets
        featureSet = cell(1, 1);
        featureSet{1} = featureSelectionOutput.(featureSetNames{j}).featureSet;
        
        idx = (j-1) * (noFolds * noParamCombs);
        fullParams(idx+1:idx+noFolds*noParamCombs, 4) = repmat(featureSet, noFolds * noParamCombs, 1);
    end
end

function [results] = trainAndEvaluateModels(X, y, model, parameters)
    N = size(parameters, 1);
    
    results = cell(N, 1);
    
    parfor n = 1:N
        params = parameters(n, :);
        
        trainingIdx = params{1};
        testIdx = params{2};
        hyperParams = params{3};
        featureSet = params{4};
    
        trainingData = X(trainingIdx, :);
        trainingLabels = y(trainingIdx);
    
        testData = X(testIdx, :);
        testLabels = y(testIdx);
    
        [trainingData, trainingLabels] = borderlineSmote(trainingData, trainingLabels, 0, 9);
    
        classifier = trainClassifier(trainingData(:, featureSet), trainingLabels, model.type, hyperParams, 1);
        evaluation = classifier.classify(testData(:, featureSet), testLabels);
    
        results{n} = evaluation;
    end
end