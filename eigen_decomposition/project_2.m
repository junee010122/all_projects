% Load the dataset
data = load('GMD_Train_S24.dat');

% Initialize constants and parameters
K = 3; % Number of mixture components
D = size(data, 2); % Dimensionality of data
max_iters = 20; % Maximum number of iterations

% Initialize the model parameters
% Initialize mixture weights
pi_k = ones(1, K) / K;


Sigma_k = repmat(eye(D), [1, 1, K]);

mu_k = [0, 0; -1, -1; 4, 0];

log_likelihood = zeros(max_iters, 1);

for iter = 1:max_iters

    r_ik = zeros(size(data, 1), K);
    for k = 1:K
        r_ik(:, k) = pi_k(k) * mvnpdf(data, mu_k(k, :), Sigma_k(:, :, k));
    end
    r_ik = r_ik ./ sum(r_ik, 2);

   
    log_likelihood(iter) = sum(log(sum(r_ik .* pi_k, 2)));
    

    N_k = sum(r_ik, 1);
    pi_k = N_k / sum(N_k);
    for k = 1:K
        mu_k(k, :) = sum(data .* r_ik(:, k), 1) / N_k(k);
        Sigma_k(:, :, k) = (data - mu_k(k, :))' * diag(r_ik(:, k)) * (data - mu_k(k, :)) / N_k(k);
    end
end

figure;
plot(1:max_iters, log_likelihood, 'o-');
xlabel('Iterations');
ylabel('Log-Likelihood');
title('Log-Likelihood vs. Iterations');
grid on;

r_ik_final = zeros(size(data, 1), K);
for k = 1:K
    r_ik_final(:, k) = pi_k(k) * mvnpdf(data, mu_k(k, :), Sigma_k(:, :, k));
end
r_ik_final = r_ik_final ./ sum(r_ik_final, 2);


[~, cluster_indices] = max(r_ik_final, [], 2);


color_map = lines(K); % Use different colors for each cluster


figure;
for k = 1:K
    cluster_data = data(cluster_indices == k, :);
    scatter(cluster_data(:, 1), cluster_data(:, 2), 20, color_map(k, :), 'filled');
    hold on;
end


xlabel('Feature 1');
ylabel('Feature 2');
title('2D Scatter Plot of Data Samples Assigned to Clusters');
legend({'Cluster 1', 'Cluster 2', 'Cluster 3'}, 'Location', 'best');
grid on;



data = load('HalfMoon_S24.txt');
X = data(:,1:2);


sigma2 = 1;
epsilon = 0.5;


W = zeros(size(X,1), size(X,1));
for i = 1:size(X,1)
    for j = 1:size(X,1)
        if norm(X(i,:) - X(j,:)) <= epsilon
            W(i,j) = exp(-norm(X(i,:) - X(j,:))^2 / (2*sigma2));
        end
    end
end


D = diag(sum(W, 2));


L = D - W;


[V, E] = eig(L, D);


figure;
plot(diag(E));
xlabel('Eigenvalue index');
ylabel('Eigenvalue');


[~, idx] = sort(diag(E));
y = V(:, idx(2));


figure;
plot(y);
xlabel('Data Sample Index');
ylabel('Eigenvector Value');


threshold = mean(y);


cluster1 = X(y <= threshold, :);
cluster2 = X(y > threshold, :);


figure;
scatter(cluster1(:,1), cluster1(:,2), 'r', 'filled');
hold on;
scatter(cluster2(:,1), cluster2(:,2), 'b', 'filled');
hold off;
xlabel('Feature 1');
ylabel('Feature 2');
legend('Cluster 1', 'Cluster 2');



data = load('GMD_Train_S24.dat');


K = 3; 
N = size(data, 1); 
d = size(data, 2); 
iterations = 20; 


pi_k = ones(1, K) ./ K; 
mu_k = [0 0; -1 -1; 4 0]; 
sigma_k = repmat(eye(d), [1 1 K]); 


log_likelihood = zeros(iterations, 1);


for iter = 1:iterations
   
    gamma_nk = zeros(N, K);
    for k = 1:K
        gamma_nk(:, k) = mvnpdf(data, mu_k(k, :), sigma_k(:, :, k)) * pi_k(k);
    end
    gamma_nk = gamma_nk ./ sum(gamma_nk, 2);

 
    Nk = sum(gamma_nk, 1);
    pi_k = Nk / N;
    mu_k = (gamma_nk' * data) ./ Nk';
    for k = 1:K
        data_centered = data - mu_k(k, :);
        sigma_k(:, :, k) = (data_centered' * (gamma_nk(:, k) .* data_centered)) ./ Nk(k);
    end

    
    log_likelihood(iter) = sum(log(sum(gamma_nk * pi_k')));

   
    if iter > 1 && abs(log_likelihood(iter) - log_likelihood(iter-1)) < 1e-6
        break;
    end
end


figure;
plot(1:iter, log_likelihood(1:iter), '-o');
xlabel('Iteration');
ylabel('Log-Likelihood');
title('Log-Likelihood vs Iteration');


[~, k_star] = max(gamma_nk, [], 2);


figure;
scatter(data(k_star == 1, 1), data(k_star == 1, 2), 'r');
hold on;
scatter(data(k_star == 2, 1), data(k_star == 2, 2), 'b');
scatter(data(k_star == 3, 1), data(k_star == 3, 2), 'g');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');
xlabel('Feature 1');
ylabel('Feature 2');
title('Data samples assigned to each cluster');
hold off;


fprintf('Final Mixture Weights: %s\n', num2str(pi_k));
fprintf('Final Mean Vectors: %s\n', num2str(mu_k));
for k = 1:K
    fprintf('Final Covariance Matrix for Cluster %d:\n', k);
    disp(sigma_k(:, :, k));
end

