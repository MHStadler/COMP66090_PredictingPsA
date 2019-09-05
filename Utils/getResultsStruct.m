function [ resultsStruct ] = getResultsStruct( results )

resultsStruct = cellfun(@(x) x.createResultStruct(), results);

end