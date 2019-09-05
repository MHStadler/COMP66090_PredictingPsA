function [ hyperParameterCombinations ] = buildHyperParameterCombinations(hyperParameters) 
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

    hyperParameterCombinations = table2struct(tbl);
end