classdef RbfSvmClassifier < IClassifier
    properties(Access = private)
        gamma,
        penalty
    end
    
    methods
        function obj = RbfSvmClassifier(X, y, hyperParameters, positiveClass)
            kernelScale = 1 / sqrt(hyperParameters.GAMMA);
            C = hyperParameters.C;
            
            svm = fitcsvm(X, y, 'KernelFunction', 'gaussian', ...
                'KernelScale', kernelScale, 'BoxConstraint', C);
            model = fitSVMPosterior(svm);
            
            obj@IClassifier(model, positiveClass);
            
            obj.gamma = hyperParameters.GAMMA;
            obj.penalty = hyperParameters.C;
        end
    end
end