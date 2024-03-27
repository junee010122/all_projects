% Load the dataset
data = load('HalfMoon_S24.txt');

distances = squareform(pdist(data));

sigma_squared = 1;
W = exp(-distances.^2 / (2*sigma_squared));

D = diag(sum(W));


L = D - W;


[V, E] = eig(L, D);


eigenvalues = diag(E);
[eigenvalues_sorted, index_sorted] = sort(eigenvalues);


figure;
plot(eigenvalues_sorted, '.-');
title('Eigenvalues Plot');
xlabel('Index');
ylabel('Eigenvalue');


y = V(:, index_sorted(2));


figure;
plot(y, '.-');
title('Eigenvector Plot');
xlabel('Index');
ylabel('Eigenvector Value');

threshold = mean(y);
clusters = y > threshold;


figure;
scatter(data(:, 1), data(:, 2), 20, clusters);
title('Scatter Plot with Clusters');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(lines);




data = load('TwoSquaresThreeCircles_S24.dat');


sigma_squared = 0.5;
dist_threshold = 1;


similarity_matrix = zeros(size(data,1), size(data,1));
for i = 1:size(data,1)
    for j = 1:size(data,1)
        similarity_matrix(i,j) = exp(-(norm(data(i,:)-data(j,:))^2)/(2*sigma_squared));
    end
end


D = diag(sum(similarity_matrix,2));
L = D - similarity_matrix;


[V,E] = eig(L,D);
eigenvalues = diag(E);


[sorted_eigenvalues, idx] = sort(eigenvalues);


figure;
plot(1:length(sorted_eigenvalues), sorted_eigenvalues, '-o');
xlabel('Eigenvalue index');
ylabel('Eigenvalue');
title('Eigenvalues of Laplacian');


stacked_eigenvectors = V(:,idx(2:5));

 figure;
 
 plot(stacked_eigenvectors(:,1), 'r');
 xlabel('Data index');
 ylabel('Value');
 title('2nd Eigenvector');
 
 subplot(2, 2, 2);
 plot(stacked_eigenvectors(:,2), 'g');
 xlabel('Data index');
 ylabel('Value');
 title('3rd Eigenvector');
 
 subplot(2, 2, 3); 
 plot(stacked_eigenvectors(:,3), 'b');
 xlabel('Data index');
 ylabel('Value');
 title('4th Eigenvector');
 
 subplot(2, 2, 4); 
 plot(stacked_eigenvectors(:,4), 'm');
 xlabel('Data index');
 ylabel('Value');
 title('5th Eigenvector');
 

 K = 5;
 initial_centroids = [stacked_eigenvectors(150,:); stacked_eigenvectors(450,:); stacked_eigenvectors(800,:); stacked_eigenvectors(1100,:); stacked_eigenvectors(1300,:)];
 
 [idx, centroids] = kmeans(stacked_eigenvectors, K, 'Start', initial_centroids);
 
 clusters = zeros(size(data,1), 1);
 for i = 1:K
     clusters(idx == i) = i;
 end

figure;
gscatter(data(:,1), data(:,2), clusters);
xlabel('Feature 1');
ylabel('Feature 2');
title('2D scatter plot with assigned clusters');



data = load('TwoSquaresThreeCircles_S24.dat');

K = 5;
[idx, centroids] = kmeans(data, K, 'Start', data([150 450 800 1100 1300],:));


figure;
gscatter(data(:,1), data(:,2), idx);
hold on;
plot(centroids(:,1), centroids(:,2), 'x', 'MarkerSize', 15, 'LineWidth', 2, 'MarkerEdgeColor', 'k');
xlabel('Feature 1');
ylabel('Feature 2');
title('2D scatter plot with assigned clusters and centroids');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Centroids');