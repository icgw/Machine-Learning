function Z = LLE(D, k, d)
% Locally Linear Embedding 局部线性嵌入
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

%% 计算线性重构的系数
W = zeros(k, m);
for i = 1:m
    id = neighbors(:, i);
    z = D(:, i) - D(:, id);
    % 计算线性重构的系数
    C = z' * z;
    % 添加一个正则项，以避免 C 的奇异性导致计算不稳定
    C = C + eye(k,k) * tol * trace(C);
    W(:, i) = C \ ones(k, 1);
    W(:, i) = W(:, i) ./ sum(W(:, i));
end

%% 计算投影矩阵
M = eye(m);
for i = 1:m
    w = W(:, i);
    id = neighbors(:, i);
    M(i, id) = M(i, id) - w';
    M(id, i) = M(id, i) - w;
    M(id, id) = M(id, id) + w * w';
end
[Z, ~] = eigs(M, d + 1, 'smallestabs');
Z = Z(:, 2:d+1)' * sqrt(m);
end