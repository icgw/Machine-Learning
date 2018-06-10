function Z = LLE(D, k, d)
% Locally Linear Embedding �ֲ�����Ƕ��
%    ���룺������ D�����ڲ��� k����ά�ռ�ά�� d
%    ����������� D �ڵ�ά�ռ��ͶӰ
%%
[~, m] = size(D);
% ����������ϵ��
tol = 1e-3;
%% ���� k-���ڵ�
D2         = sum(D .* D);
dist       = repmat(D2, m, 1) + repmat(D2', 1, m) - 2 * (D' * D);
[~, index] = mink(dist, k + 1);
neighbors  = index(2:k+1, :);

%% ���������ع���ϵ��
W = zeros(k, m);
for i = 1:m
    id = neighbors(:, i);
    z = D(:, i) - D(:, id);
    % ���������ع���ϵ��
    C = z' * z;
    % ���һ��������Ա��� C �������Ե��¼��㲻�ȶ�
    C = C + eye(k,k) * tol * trace(C);
    W(:, i) = C \ ones(k, 1);
    W(:, i) = W(:, i) ./ sum(W(:, i));
end

%% ����ͶӰ����
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