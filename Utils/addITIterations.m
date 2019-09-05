function [ newFeatureSelectionOutput ] = addITIterations( featureSelectionOutput )

newFeatureSelectionOutput = struct();

values = [5, 10, 25, 50, 100];

structFields = fieldnames(featureSelectionOutput);
noStructFields = numel(structFields);

noValues = numel(values);
for i = 1:noStructFields
    name = structFields{i};
    
    result = featureSelectionOutput.(name);
    
    if(strcmp(result.type, 'FEAST'))
        featureSet = result.featureSet;
    
        for n = 1:noValues
            value = values(n);
        
            newName = strcat(name, '_', num2str(value));
        
            newFeatureSelectionOutput.(newName).featureSet = featureSet(1:value);
        end
    else
        newFeatureSelectionOutput.(name) = result;
    end
end

end