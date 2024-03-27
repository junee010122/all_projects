% Define your 2D dataset
dataset = [
    2 5;
    6 4;
    5 3;
    2 2;
    1 4;
    5 2;
    3 3;
    2 3;
];

% Calculate the pairwise Euclidean distances
distances = pdist(dataset);

% Create the distance matrix D (upper triangle part)
n = size(dataset, 1);
D = zeros(n);
D(triu(true(n), 1)) = distances;

% Display the distance matrix
D


% Carry out agglomerative clustering for complete linkage
Z_complete = linkage(D, 'complete');

% Plot dendrogram for complete linkage
figure;
subplot(1, 3, 1);
dendrogram(Z_complete);
title('Complete Linkage');

% Carry out agglomerative clustering for average linkage
Z_average = linkage(D, 'average');

% Plot dendrogram for average linkage
subplot(1, 3, 2);
dendrogram(Z_average);
title('Average Linkage');

% Carry out agglomerative clustering for Ward's method
Z_ward = linkage(X, 'ward');

% Plot dendrogram for Ward's method
subplot(1, 3, 3);
dendrogram(Z_ward);
title('Ward Method');