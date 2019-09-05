function [featureSelectionOutput] = performFeatureSelection(X, y, params )

featureSelectionOutput = struct();

featureSelectionParams = params.FeatureSelection;

noCVFolds = featureSelectionParams.NoCVFolds;

for i = 1:numel(featureSelectionParams.Techniques)
    technique = featureSelectionParams.Techniques(i);
    
    fprintf('Processing technique: %s\n', technique.name);
    
    type = technique.type;
    if(strcmp(type, 'elasticNet'))
        fprintf('Starting elasticNet FeatureSelection!\n');
        
        featureSelectionOutput = selectElasticNetFeatures(featureSelectionOutput, X, y, technique, noCVFolds);
        
        fprintf('Finished Lasso!\n');
    elseif(strcmp(type, 'FEAST'))
        fprintf('Starting feast procedure!\n');
        
        featureSelectionOutput = selectFeastFeatures(featureSelectionOutput, X, y, technique, noCVFolds);
        
        fprintf('Finished feast!\n');
    else
        fprintf('No handler for technique!\n');
    end 
end

toc;

end

function [ featureSelectionOutput ] = selectElasticNetFeatures(featureSelectionOutput, X, y, technique, noCVFolds)
    alphaValues = technique.params.alpha;
    noAlphaValues = numel(alphaValues);
    
    for i = 1:noAlphaValues
        alpha = alphaValues(i);
        
        elasticNetFeatureSet = elasticNetCVFeatureSelection(X, y, alpha, noCVFolds);
              
        elasticNetResult = struct();
        elasticNetResult.featureSet = elasticNetFeatureSet;
        elasticNetResult.type = technique.type;
        elasticNetResult.alpha = alpha;
        
        featureSelectionOutput.(strcat('ElasticNet_', num2str(i))) = elasticNetResult;
    end
end


function [ featureSelectionOutput ] = selectFeastFeatures(featureSelectionOutput, X, y, technique, noCVFolds)
    noFeatures = technique.params.noFeatures;

    feastTechniques = technique.params.technique;
    noTechniques = numel(feastTechniques);
    
    for i = 1:noTechniques
        feastTechnique = feastTechniques{i};
        
        feastFeatureSet = feastCVFeatureSelection(X, y, feastTechnique, noFeatures, noCVFolds);
              
        feastResult = struct();
        feastResult.featureSet = feastFeatureSet;
        feastResult.type = technique.type;
        feastResult.feastTechnique = feastTechnique;
        feastResult.noFeatures = noFeatures;
        
        
        featureSelectionOutput.(strcat('FEAST_', feastTechnique)) = feastResult;
    end
end