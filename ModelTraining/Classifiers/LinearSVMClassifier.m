classdef LinearSVMClassifier < IClassifier
    properties
        C
    end
    
    methods
        function obj = LinearSVMClassifier(X, y, hyperParameters, positiveClass) 
            C = hyperParameters.C;
            
            svm = fitcsvm(X, y, ...
                'BoxConstraint', C);
            model = fitSVMPosterior(svm);
            
            obj@IClassifier(model, positiveClass); 
            obj.C = C;
        end
    end
end