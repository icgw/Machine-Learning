function Z = LE(D, k, d)
% Laplacian Eigenmaps ������˹����ӳ��
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

%% �׺ͶȾ���ͶȾ���
W = zeros(m, m);
for i = 1:m
    id = neighbors(:, i);
    W(i, id) = dist(i, id);
    W(id, i) = dist(i, id)';
end
M = W - diag(sum(W));

% �� M ��������������㲻�ȶ�
M = M + eye(m) * tol * trace(M);

[Z, ~] = eigs(M, d + 1, 'smallestabs');
Z = Z(:, 2:d+1)' * sqrt(m);
end