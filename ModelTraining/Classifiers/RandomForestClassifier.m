classdef RandomForestClassifier < IClassifier
    properties
        noTrees,
        leafSize,
        noPredictors
    end
    
    methods
        function obj = RandomForestClassifier(X, y, hyperParameters, positiveClass) 
            noTrees = hyperParameters.RF_SIZE;
            leafSize = hyperParameters.RF_LS;
            
            if hyperParameters.RF_NP ~= 1
                noPredictors = ceil(size(X, 2) * hyperParameters.RF_NP);
            else
                noPredictors = 'all';
            end
            
            model = TreeBagger(noTrees, X, y, 'OOBPrediction','on', 'OOBPredictorImportance','on', ...
                'Method', 'classification', 'MinLeafSize', leafSize, 'NumPredictorsToSample', noPredictors);
            
            classNames = cell2mat(model.ClassNames);
            posClassIdx = find(classNames == num2str(positiveClass));
            
            obj@IClassifier(model, positiveClass, posClassIdx); 
            obj.noTrees = noTrees;
            obj.leafSize = leafSize;
            obj.noPredictors = noPredictors;
        end
        
        function [classificationResult] = classify(obj, X, expectedLabels)
            [predictedLabels, score] = obj.performClassification(X);
            
            predictedLabels = str2double(predictedLabels);
            
            classificationResult = ModelEvaluator(expectedLabels, predictedLabels, score(:, obj.posClassIdx), obj.PositiveClass);
        end
    end
end