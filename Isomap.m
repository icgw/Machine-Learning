function Z = Isomap(D, k, d)
% Isometric Mapping 等度量映射
%    输入：样本集 D，近邻参数 k，低维空间维数 d
%    输出：样本集 D 在低维空间的投影
%%
[~, m] = size(D);
% 初始化邻接矩阵
adj = inf(m, m);
for i = 1:m
    x = D(:, i);
    % 确定 xi 的 k 近邻
    dist = sqrt(sum((x - D) .* (x - D))); dist(i) = inf;
    [~, id] = mink(dist, k);
    % xi 与 k 近邻点之间的距离设置为欧式距离
    adj(i, id) = dist(id);
    adj(id, i) = dist(id);
end
% 构造数据图
G = graph(adj);
% 计算最短路径
dist = distances(G);
% 调用 MDS 算法
Z = MDS(dist, d);
end