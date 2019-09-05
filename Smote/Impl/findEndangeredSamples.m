function [ endangeredSamples ] = findEndangeredSamples(X, y, minorityClass, m) 
    minoritySamples = X(y == minorityClass, :);
    
    N = size(minoritySamples, 1);   
    
    endangeredIdx = boolean(zeros(N, 1));
    
    % Find neighbours of each minority sample
    idx = knnsearch(X, minoritySamples, 'K', m+1);
    neighbourhoodY = y(idx(:, 2:end));
    
    % Number of majority neighbours needed before sample is considered
    % endangered
    endangeredCutoff = m / 2;
    for i = 1:N
        noMajorityClassNeighbours = nnz(neighbourhoodY(i, :) ~= minorityClass);
        
        % If the sample is not noise and if more than half its neighbours
        % are of the majority class, consider it endangered
        if noMajorityClassNeighbours ~= m && noMajorityClassNeighbours >= endangeredCutoff
            endangeredIdx(i) = true;
        end
    end
    
    endangeredSamples = minoritySamples(endangeredIdx, :);
end