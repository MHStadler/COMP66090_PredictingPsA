function [ X, headerLine ] = importHugeTextFile( N, D, nMetaColumns, delim )
%importHugeTextFile Import data from a huge text file
%
%   [ X, headerLine ] = importHugeTextFile( N, D, nMetaColumns, delim )
%
%   Import data from a huge text file. N is the number of rows in the file,
%   D is the number of columns, and nMetaColumns is the number of meta data
%   columns that should be excluded from the data matrix. delim is the
%   delimiter used to separate the columns in the original file. 
%   The output X is the resulting data matrix, and headerLine contains the
%   header items for the returned data.

[databaseFileName,databasePath,~] = uigetfile('*.*','Select a Database...');
databaseFullPath = fullfile(databasePath,databaseFileName);

tic;

dataMat = readHugeFile(N, D, 1, databaseFullPath, delim);

X = zeros(N-1, D-nMetaColumns);

headerLine = dataMat{1}{1};
for i = 2:N
    x = str2double(dataMat{i}{1});
    
    X(i-1, :) = x(nMetaColumns+1:end);
end

toc;

end

function [ X ] = readHugeFile(N, D, n, path, delim)

fid = fopen(path);

k = 1;

X = cell(ceil(N / n), 1);

while ~feof(fid)
    C = textscan(fid, '%s', D * n, 'CommentStyle', '##', 'Delimiter', delim, 'HeaderLines', 0);
    
    if(~isempty(C))
        X{k} = C;
        k = k + 1;
    end
end

fclose(fid);

end