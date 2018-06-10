function Z = LE(D, k, d)
% Laplacian Eigenmaps 拉普拉斯特征映射
%    输入：样本集 D，近邻参数 k，低维空间维数 d
%    输出：样本集 D 在低维空间的投影
%%
[~, m] = size(D);
% 设置正则项系数
tol = 1e-3;
%% 计算 k-近邻点
D2         = sum(D .* D);
dist       = repmat(D2, m, 1) + repmat(D2', 1, m) - 2 * (D' * D);
[~, index] = mink(dist, k + 1);
neighbors  = index(2:k+1, :);

%% 亲和度矩阵和度矩阵
W = zeros(m, m);
for i = 1:m
    id = neighbors(:, i);
    W(i, id) = dist(i, id);
    W(id, i) = dist(i, id)';
end
M = W - diag(sum(W));

% 给 M 添加正则项，避免计算不稳定
M = M + eye(m) * tol * trace(M);

[Z, ~] = eigs(M, d + 1, 'smallestabs');
Z = Z(:, 2:d+1)' * sqrt(m);
end