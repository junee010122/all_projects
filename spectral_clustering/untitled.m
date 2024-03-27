% Step 1: Read the distance matrix
distances = dlmread('MOCityDistS24.dat');

% Step 2: Classical Multidimensional Scaling (MDS)
n = size(distances, 1);
J = eye(n) - ones(n)/n;
B = -0.5 * J * distances.^2 * J;
[V,D] = eig(B);
[d,ind] = sort(diag(D), 'descend');
V = V(:,ind);
D = D(ind,ind);
d = sqrt(D);
X = V(:,1:2)*d(1:2,1:2);

% Step 3: Determine the coordinates of Columbia (x3) and Rolla (x6)
columbia_index = 3;
rolla_index = 6;
x3 = X(columbia_index,:);
x6 = X(rolla_index,:);

% Step 4: Recompute Euclidean distances
recomputed_distance_columbia_rolla = norm(x3 - x6);
kansas_city_index = 5;
st_louis_index = 7;
recomputed_distance_kc_stl = norm(X(kansas_city_index,:) - X(st_louis_index,:));

% Step 5: Plot the map
cities = {'Branson', 'Cape Girardeau', 'Columbia', 'Jefferson City', 'Kansas City', ...
          'Rolla', 'St. Louis', 'Springfield', 'St. Joseph', 'Independence'};
figure;
scatter(X(:,1), X(:,2), 'filled');
text(X(:,1), X(:,2), cities, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
title('Map of Missouri Cities (MDS Coordinates)');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
axis equal;

% Output the results
fprintf('2.1 Columbia coordinates (x3): [%f, %f]\n', x3(1), x3(2));
fprintf('2.2 Rolla coordinates (x6): [%f, %f]\n', x6(1), x6(2));
fprintf('2.3.1 Recomputed Euclidean distance between Columbia and Rolla: %f\n', recomputed_distance_columbia_rolla);
fprintf('2.3.2 Recomputed Euclidean distance between Kansas City and St. Louis: %f\n', recomputed_distance_kc_stl);
