% Load the word-document matrix from the file
W = load('WordDocS24.dat');

% Perform SVD on the matrix
[U, S, V] = svd(W);

% Initialize variables to store approximation errors
approximation_errors = zeros(1,3);

% Rank-k approximation for k = 2, 4, 6
for k = [2, 4, 6]
    % Construct rank-k approximation
    W_approx = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
    
    % Compute the Frobenius norm squared of the difference
    approximation_errors(k/2) = norm(W - W_approx, 'fro')^2;
end

% Display the approximation errors
disp("Approximation Errors (Frobenius Norm Squared):");
disp(approximation_errors);


% Rank-4 approximation
k = 4;
W_approx = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';

% Compute inner-product similarity for document pairs
document_similarities = W_approx * W_approx';
% Set the diagonal to zero to avoid self-similarity
document_similarities(logical(eye(size(document_similarities)))) = 0;

% Find the indices of the maximum similarity value
[max_sim_doc, doc_indices] = max(document_similarities(:));
[row, col] = ind2sub(size(document_similarities), doc_indices);

% Display the most similar document pair and similarity value
fprintf('Most similar document pair: Document %d and Document %d\n', row, col);
fprintf('Similarity value: %.4f\n\n', max_sim_doc);

% Compute inner-product similarity for word pairs
word_similarities = W_approx' * W_approx;
% Set the diagonal to zero to avoid self-similarity
word_similarities(logical(eye(size(word_similarities)))) = 0;

% Find the indices of the maximum similarity value
[max_sim_word, word_indices] = max(word_similarities(:));
[row, col] = ind2sub(size(word_similarities), word_indices);

% Display the most similar word pair and similarity value
fprintf('Most similar word pair: Word %d and Word %d\n', row, col);
fprintf('Similarity value: %.4f\n', max_sim_word);

% Compute cosine similarity for document pairs
document_similarities = pdist2(W_approx', W_approx', 'cosine');

% Exclude self-similarity (set diagonal elements to a large value to exclude self-similarity)
document_similarities(logical(eye(size(document_similarities)))) = Inf;

% Find the indices of the minimum similarity value for document pairs
[min_sim_doc, doc_indices] = min(document_similarities(:));
[row_doc, col_doc] = ind2sub(size(document_similarities), doc_indices);

% Display the most similar document pair and similarity value
fprintf('Most similar document pair: Document %d and Document %d\n', row_doc, col_doc);
fprintf('Similarity value: %.4f\n\n', 1 - min_sim_doc);

% Compute cosine similarity for word pairs
word_similarities = pdist2(W_approx, W_approx, 'cosine');

% Exclude self-similarity (set diagonal elements to a large value to exclude self-similarity)
word_similarities(logical(eye(size(word_similarities)))) = Inf;

% Find the indices of the minimum similarity value for word pairs
[min_sim_word, word_indices] = min(word_similarities(:));
[row_word, col_word] = ind2sub(size(word_similarities), word_indices);

% Display the most similar word pair and similarity value
fprintf('Most similar word pair: Word %d and Word %d\n', row_word, col_word);
fprintf('Similarity value: %.4f\n', 1 - min_sim_word);
