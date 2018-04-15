function predict = softmaxRegression(sample, lambda, gamma, iters, threshold)
%SOFTMAXREGRESSION �˴���ʾ�йش˺�����ժҪ
%   ���룺sample Ϊ (n,d,c) �ľ���
%             1..c �������㣬ÿ��������ģΪ n��������Ϊ d ά����
%         lambda Ϊ���򻯲�����
%         gamma  Ϊ�ݶ��½�ʱ�Ĳ�����Ĭ��Ϊ 0.1
%         iters  Ϊ�ݶ��½�ʱ���ĵ���������Ĭ��Ϊ 5000
%         threshold Ϊ�ݶ��½�ʱ�˳�ѭ������ֵ��Ĭ��Ϊ 1e-5

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

% ��ʼ��
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
