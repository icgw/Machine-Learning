function predict = softmaxRegression(sample, lambda, gamma, iters, threshold)
%SOFTMAXREGRESSION - softmax 回归模型
%   输入：sample    - 为 (m,d,c) 的矩阵；
%             1..c 类样本点，每类样本规模为 d，样本点为 m 维向量。
%         lambda    - 为正则化参数；
%         gamma     - 为梯度下降时的步长，默认为 0.1
%         iters     - 为梯度下降时最大的迭代步数，默认为 5000
%         threshold - 为梯度下降时退出循环的阈值，默认为 1e-5
%   输出：predict(X) - 预测函数
%                     * 参数为测试集 X，测试点为列向量
%                     * 输出为列向量，第 i 行的值对应与 X(:, i) 测试点的分类。
%%
if nargin == 2
    gamma = 0.1;
    iters = 5000;
    threshold = 1e-5;
end

[m, d, c] = size(sample);
n = d * c;
P = @(j, X, W) exp(W(:, j)' * X) ./ sum(exp(W' * X));

pW = @(j, X, W) (sum(X(:, :) .* P(j, X(:, :), W), 2) - sum(X(:,:,j), 2))...
    ./ n + 2 .* lambda .* W(:, j);

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
predict = @(X) predictLabel(P, c, X, W);
end

function lw = loss(n, c, lambda, P, X, W)
tp = log(P(1, X(:, :, 1), W));
for i = 2:c
    tp = tp + log(P(i, X(:, :, i), W));
end
lw = -sum(tp) ./ n + lambda .* sum(W(:) .* W(:));
end

function label = predictLabel(P, c, X, W)
[~, n] = size(X);
prob = zeros(c, n);
for j = 1:c
    prob(j, :) = P(j, X, W);
end
[label, ~] = find(prob == max(prob));
end
