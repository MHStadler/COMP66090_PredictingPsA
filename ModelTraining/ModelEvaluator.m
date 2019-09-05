classdef ModelEvaluator
    properties
        ExpectedLabels,
        PredictedLabels,
        Score,
        PositiveClass,
        TP,
        FP,
        TN,
        FN,
        ACCURACY,
        KAPPA,
        ERROR_RATE,
        ROC_X,
        ROC_Y,
        AUC,
        PRECISION,
        SENSITIVITY,
        SPECIFICITY,
        F_MEASURE
    end
    
    methods
        function obj = ModelEvaluator(expectedLabels, predictedLabels, score, positiveClass)
            obj.ExpectedLabels = expectedLabels;
            obj.PredictedLabels = predictedLabels;
            obj.Score = score;
            obj.PositiveClass = positiveClass;
            
            obj.TP = nnz(obj.PredictedLabels == obj.ExpectedLabels & obj.ExpectedLabels == positiveClass);
            obj.FP = nnz(obj.PredictedLabels ~= obj.ExpectedLabels & obj.ExpectedLabels ~= positiveClass);
            
            obj.TN = nnz(obj.PredictedLabels == obj.ExpectedLabels & obj.ExpectedLabels ~= positiveClass);
            obj.FN = nnz(obj.PredictedLabels ~= obj.ExpectedLabels & obj.ExpectedLabels == positiveClass);
        end
        
        function [accuracy, obj] = calcAccuracy(obj)
            accuracy = (obj.TP + obj.TN) / size(obj.ExpectedLabels, 1);
            
            obj.ACCURACY = accuracy;
        end
        
        function [kappa, obj] = calcKappa(obj)
            acc = obj.calcAccuracy();
            
            ranAcc = ((obj.TP + obj.FN) * (obj.TP + obj.FP) + (obj.TN + obj.FP) * (obj.TN + obj.FN)) / size(obj.ExpectedLabels, 1)^2;
            kappa = (acc - ranAcc) / (1 - ranAcc);
            
            obj.KAPPA = kappa;
        end
        
        function [errorRate, obj] = calcErrorRate(obj)
            errorRate = (obj.FP + obj.FN) / size(obj.ExpectedLabels, 1);
            
            obj.ERROR_RATE = errorRate;
        end
        
        function [auc, x, y, obj] = calcAuc(obj)
            [x, y, ~, auc] = perfcurve(obj.ExpectedLabels, obj.Score, obj.PositiveClass);
            
            obj.ROC_X = x;
            obj.ROC_Y = y;
            obj.AUC = auc;
        end
        
        function [precision, obj] =  calcPrecision(obj) 
            precision = obj.TP / (obj.TP + obj.FP);
            
            obj.PRECISION = precision;
        end
        
        function [sensitivity, specificity, obj] = calcSensitivityAndSpecificity(obj)
            sensitivity = obj.TP / (obj.TP + obj.FN);
            specificity = obj.TN / (obj.TN + obj.FP);
            
            obj.SENSITIVITY = sensitivity;
            obj.SPECIFICITY = specificity;
        end
        
        function [fMeasure, obj] = calcFMeasure(obj)
            fMeasure = 2 * obj.TP / (2 * obj.TP + obj.FN + obj.FP);
            
            obj.F_MEASURE = fMeasure;
        end
        
        function result = createResultStruct(obj) 
            result = struct();
            
            props = properties(obj);
            for i = 1:length(props)
                property = props{i};
                
                if ~isempty(obj.(property))
                    result.(property) = obj.(property);
                end
            end
        end
    end
end