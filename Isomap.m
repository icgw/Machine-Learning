function Z = Isomap(D, k, d)
% Isometric Mapping �ȶ���ӳ��
%    ���룺������ D�����ڲ��� k����ά�ռ�ά�� d
%    ����������� D �ڵ�ά�ռ��ͶӰ
%%
[~, m] = size(D);
% ��ʼ���ڽӾ���
adj = inf(m, m);
for i = 1:m
    x = D(:, i);
    % ȷ�� xi �� k ����
    dist = sqrt(sum((x - D) .* (x - D))); dist(i) = inf;
    [~, id] = mink(dist, k);
    % xi �� k ���ڵ�֮��ľ�������Ϊŷʽ����
    adj(i, id) = dist(id);
    adj(id, i) = dist(id);
end
% ��������ͼ
G = graph(adj);
% �������·��
dist = distances(G);
% ���� MDS �㷨
Z = MDS(dist, d);
end