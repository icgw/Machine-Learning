function predict = softmaxRegression(sample, lambda, gamma, iters, threshold)
%SOFTMAXREGRESSION 此处显示有关此函数的摘要
%   输入：sample 为 (n,d,c) 的矩阵；
%             1..c 类样本点，每类样本规模为 n，样本点为 d 维向量
%         lambda 为正则化参数；
%         gamma  为梯度下降时的步长，默认为 0.1
%         iters  为梯度下降时最大的迭代步数，默认为 5000
%         threshold 为梯度下降时退出循环的阈值，默认为 1e-5

if nargin == 2
    gamma = 0.1;
    iters = 5000;
    threshold = 1e-5;
end

[m, d, c] = size(sample);
n = d * c;
P = @(j, X, W) exp(W(:, j)' * X) ./ sum(exp(W' * X));

pW = @(j, X, W) (sum(X(:, :) .* P(j, X(:, :), W), 2) - sum(X(:,:,j), 2)) ./ n + ...
    2 .* lambda .* W(:, j);

% 初始化
W  = rand(m, c);
e = loss(n, c, lambda, P, sample, W);

dW = zeros(m, c);
for i = 1:iters
    for j = 1:c
        dW(:, j) = pW(j, sample, W);
    end
    W = W - gamma .* dW;
    en = loss(n, c, lambda, P, sample, W);
    if abs(en - e) < threshold
        break
    end
    e = en;
end
predict = @(j, x) P(j, x, W); 
W
end

function lw = loss(n, c, lambda, P, X, W)
tp = log(P(1, X(:, :, 1), W));
for i = 2:c
    tp = tp + log(P(i, X(:, :, i), W));
end
lw = -sum(tp) ./ n + lambda .* sum(W(:) .* W(:));
end
