
%%% #1
X = [2, 5; 6, 4; 5, 3; 2, 2; 1, 4; 5, 4; 3, 3; 2, 3; 2, 4; 8, 2; 9, 2; 10, 2; 11, 2; 10, 3; 9, 1];


order = [1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15];


theta = 3.0;

q = 7;

cluster_representatives = [];

cluster_indices = cell(1, q);

num_clusters = 0;



for i = 1:length(order)
    data_point = X(order(i), :);
    if isempty(cluster_representatives)
        cluster_representatives = [cluster_representatives; data_point];
        num_clusters = num_clusters + 1;
        cluster_indices{num_clusters} = order(i);
    else
        distances = pdist2(data_point, cluster_representatives);

    min_distance = min(distances);
    min_cluster_index = find(distances == min_distance);

    if min_distance <= theta && num_clusters <= q

        cluster_index = min_cluster_index(1);
        cluster_indices{cluster_index} = [cluster_indices{cluster_index}, order(i)];
        cluster_representatives(cluster_index, :) = mean(X(cluster_indices{cluster_index}, :), 1);
        fprintf('CLUSTER REPRESENTATIVES %.2f\n', cluster_representatives)
        fprintf('CLUSTERS: %.2f\n', cluster_index);
    elseif num_clusters < q
        cluster_representatives = [cluster_representatives; data_point];
        num_clusters = num_clusters + 1;
        cluster_indices{num_clusters} = order(i);
    end
    end
end

disp('Cluster Representatives:');
disp(cluster_representatives);
disp('Cluster Indices:');
disp(cluster_indices);

figure;
hold on;
colors = hsv(num_clusters);
%data
for i = 1:num_clusters
scatter(X(cluster_indices{i}, 1), X(cluster_indices{i}, 2), [], colors(i, :), 'filled');
end
%clusters
for i = 1:num_clusters
scatter(cluster_representatives(i, 1), cluster_representatives(i, 2), 100, 'x', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
end
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroid 1', 'Centroid 2', 'Centroid 3');
xlabel('X-axis');
ylabel('Y-axis');

title('BSAS Clustering with Cluster Centroids');
print(gcf, '-dpng', 'bsas_clustering_plot.png');
hold off;
bsas_clustering_plot = imread("/MATLAB Drive/bsas_clustering_plot.png");
hold off;



%%% #2
% Case A
K_A = 2;
initial_centroids_A = [X(1, :); X(5, :)];

% Case B
K_B = 3;
initial_centroids_B = [X(1, :); X(5, :); X(9, :)];

% Perform k-means clustering
[cluster_indices_A, centroids_A] = kmeans(X, K_A, 'Start', initial_centroids_A, 'MaxIter', 1000, 'Display', 'final');
fprintf('dentroids_A: %.2f\n', centroids_A)
[cluster_indices_B, centroids_B] = kmeans(X, K_B, 'Start', initial_centroids_B, 'MaxIter', 1000, 'Display', 'final');
figure;
% Plot the results
figure;

% Plot Case A
subplot(1, 2, 1);
scatter(X(cluster_indices_A == 1, 1), X(cluster_indices_A == 1, 2), 50, 'filled', 'MarkerFaceColor', 'y');
hold on;
scatter(X(cluster_indices_A == 2, 1), X(cluster_indices_A == 2, 2), 50, 'filled', 'MarkerFaceColor', 'k');
scatter(centroids_A(:, 1), centroids_A(:, 2), 100, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 2);
title('K-Means Clustering (K=2) - Case A');
xlabel('X-axis');
ylabel('Y-axis');
legend('Cluster 1 (Data Points)', 'Cluster 2 (Data Points)', 'Centroids');
hold off;

% Plot Case B
subplot(1, 2, 2);
scatter(X(cluster_indices_B == 1, 1), X(cluster_indices_B == 1, 2), 50, 'filled', 'MarkerFaceColor', 'y');
hold on;
scatter(X(cluster_indices_B == 2, 1), X(cluster_indices_B == 2, 2), 50, 'filled', 'MarkerFaceColor', 'k');
scatter(X(cluster_indices_B == 3, 1), X(cluster_indices_B == 3, 2), 50, 'filled', 'MarkerFaceColor', 'b');
scatter(centroids_B(:, 1), centroids_B(:, 2), 100, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 2);
title('K-Means Clustering (K=3) - Case B');
xlabel('X-axis');
ylabel('Y-axis');
legend('Cluster 1 (Data Points)', 'Cluster 2 (Data Points)', 'Cluster 3 (Data Points)', 'Centroids');
hold off;

%%% 5
X = [
    2 5;6 4;5 3;2 2;1 4;5 2;3 3;2 3];

D = squareform(pdist(X));
Z_complete = linkage(D, 'complete');
Z_average = linkage(D, 'average');
Z_ward = linkage(X, 'ward');

figure;
subplot(1,3,1);
dendrogram(Z_complete);
title('Complete Linkage');

subplot(1,3,2);
dendrogram(Z_average);
title('Average Linkage');

subplot(1,3,3);
dendrogram(Z_ward);
title('Ward Linkage');


thresh_complete = Z_complete(end - 6, 3);
thresh_average = Z_average(end - 6, 3);
thresh_ward = Z_ward(end - 6, 3);
fprintf('Dissimilarity level for R7 (Complete Linkage): %.2f\n', thresh_complete);
fprintf('Dissimilarity level for R7 (Average Linkage): %.2f\n', thresh_average);
fprintf('Dissimilarity level for R7 (Ward Linkage): %.2f\n', thresh_ward);

num_clusters_complete = numel(unique(cluster(Z_complete, 'maxclust', 1)));
num_clusters_average = numel(unique(cluster(Z_average, 'maxclust', 1)));
num_clusters_ward = numel(unique(cluster(Z_ward, 'maxclust', 1)));
fprintf('Number of clusters using longest survival time criterion (Complete Linkage): %d\n', num_clusters_complete);
fprintf('Number of clusters using longest survival time criterion (Average Linkage): %d\n', num_clusters_average);
fprintf('Number of clusters using longest survival time criterion (Ward Linkage): %d\n', num_clusters_ward);