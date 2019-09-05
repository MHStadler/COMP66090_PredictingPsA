classdef PolynomialSvmClassifier < IClassifier
    properties
        C,
        d
    end
    
    methods
        function obj = PolynomialSvmClassifier(X, y, hyperParameters, positiveClass) 
            C = hyperParameters.C;
            d = hyperParameters.d;
            
            svm = fitcsvm(X, y, KernelFunction', 'polynomial', ...
                'PolynomialOrder', d, 'BoxConstraint', C);
            model = fitSVMPosterior(svm);
            
            obj@IClassifier(model, positiveClass); 
            obj.C = C;
            obj.d = d;
        end
    end
end