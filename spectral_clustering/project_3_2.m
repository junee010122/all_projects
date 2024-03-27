% Step 1: Load the distance matrix
distance_matrix = load('MOCityDistS24.dat');

% Step 2: Apply MDS
% Square the distances to obtain dissimilarities
dissimilarities = distance_matrix .^ 2;

% Double centering
n = size(dissimilarities, 1);
J = eye(n) - ones(n) / n;
B = -0.5 * J * dissimilarities * J;

% Eigendecomposition
[V, D] = eig(B);
[~, idx] = sort(diag(D), 'descend');
eigenvalues = diag(D(idx, idx));
eigenvectors = V(:, idx);

% Step 3: Extract coordinates
% Columbia is the third city
x3 = eigenvectors(3, 1:2)';
% Rolla is the sixth city
x6 = eigenvectors(6, 1:2)';
% Kansas City is the fifth city
x5 = eigenvectors(5, 1:2)';
% Saint Louis is the seventh city
x7 = eigenvectors(7, 1:2)';

% Since the first components of the two eigenvectors in V+ should be negative, multiply them by -1
if x3(1) > 0
    x3 = -x3;
end
if x6(1) > 0
    x6 = -x6;
end    
if x5(1) > 0
    x5 = -x5;
end
if x7(1) > 0
    x7 = -x7;
end

% Display results
fprintf('Coordinates of Columbia (x3): [%f, %f]\n', x3(1), x3(2));
fprintf('Coordinates of Rolla (x6): [%f, %f]\n', x6(1), x6(2));
fprintf('Coordinates of KC (x5): [%f, %f]\n', x5(1), x5(2));
fprintf('Coordinates of STL (x7): [%f, %f]\n', x7(1), x7(2));

% Compute the Euclidean distance between Columbia and Rolla
distance_Columbia_Rolla = sqrt(sum((x3 - x6).^2));

% Compute the Euclidean distance between Kansas City and St. Louis
distance_KansasCity_StLouis = sqrt(sum((x5 - x7).^2));

% Display results
fprintf('Recomputed Euclidean distance between Columbia and Rolla: %.2f', distance_Columbia_Rolla);
fprintf('Recomputed Euclidean distance between Kansas City and St. Louis: %.2f', distance_KansasCity_StLouis);




% Load the driving distances between each pair of cities
distance_matrix = load('MOCityDistS24.dat');

% Perform MDS to derive coordinates
% Square the distances to obtain dissimilarities
dissimilarities = distance_matrix .^ 2;

% Double centering
n = size(dissimilarities, 1);
J = eye(n) - ones(n) / n;
B = -0.5 * J * dissimilarities * J';

% Eigendecomposition
[V, D] = eig(B);
[~, idx] = sort(diag(D), 'descend');
eigenvalues = diag(D(idx, idx));
eigenvectors = V(:, idx);

% Extract coordinates
estimated_coordinates = eigenvectors(:, 1:2);

% Given city names
city_names = {'Branson', 'Cape Girardeau', 'Columbia', 'Jefferson City', 'Kansas City', ...
    'Rolla', 'St. Louis', 'Springfield', 'St. Joseph', 'Independence'};

% Plot the map
figure;
scatter(estimated_coordinates(:,1), estimated_coordinates(:,2), 'filled');
hold on;

% Annotate each point with city name
for i = 1:numel(city_names)
    text(estimated_coordinates(i,1), estimated_coordinates(i,2), city_names{i}, ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

% Set plot title and axis labels
title('Map of Ten Cities in Missouri (MDS Reconstructed Coordinates)');
xlabel('X Coordinate');
ylabel('Y Coordinate');
grid on;
axis equal;

% Save the plot as a PNG file
saveas(gcf, 'map_ten_cities.png');

