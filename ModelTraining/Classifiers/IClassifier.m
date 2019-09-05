classdef IClassifier
    properties
        Model,
        PositiveClass
    end
    
    properties(Access = private)
        posClassIdx
    end
    
    methods
        function obj = IClassifier(model, positiveClass)
            obj.Model = model;
            obj.PositiveClass = positiveClass;
            obj.posClassIdx = find(model.ClassNames == obj.PositiveClass);
        end
        
        function [classificationResult] = classify(obj, X, expectedLabels)
            [predictedLabels, score] = obj.performClassification(X);
            
            classificationResult = ModelEvaluator(expectedLabels, predictedLabels, score(:, obj.posClassIdx), obj.PositiveClass);
        end
    end
    
    methods(Access = protected)
        function [predictedLabels, score] = performClassification(obj, X)
            [predictedLabels, score] = obj.Model.predict(X);
        end
    end
end