% Load the dataset
data = load('HalfMoon_S24.txt');

% Compute the distance matrix based on Euclidean distance
% Use 'squareform' to convert the distance vector into a matrix
distances = squareform(pdist(data));

% Compute the graph affinity matrix using Gaussian similarity function
sigma_squared = 1;
W = exp(-distances.^2 / (2*sigma_squared));

% Compute the degree matrix
D = diag(sum(W));

% Compute the unnormalized graph Laplacian matrix
L = D - W;

% Perform eigen-decomposition
[V, E] = eig(L, D);

% Extract the eigenvalues and sort them in ascending order
eigenvalues = diag(E);
[eigenvalues_sorted, index_sorted] = sort(eigenvalues);

% Plot eigenvalues
%figure;
%plot(eigenvalues_sorted, '.-');
%title('Eigenvalues Plot');
%xlabel('Index');
%ylabel('Eigenvalue');

% Extract the eigenvector corresponding to the second smallest eigenvalue
y = V(:, index_sorted(2));

% Plot the eigenvector y
%figure;
%plot(y, '.-');
%title('Eigenvector Plot');
%xlabel('Index');
%ylabel('Eigenvector Value');

% Partition the eigenvector
threshold = mean(y);
clusters = y > threshold;

% Scatter plot of data samples with assigned clusters
figure;
scatter(data(:, 1), data(:, 2), 20, clusters);
title('Scatter Plot with Clusters');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(lines);

