function Z = MDS(D, d)
%Multiple Dimensional Scaling 多维度缩放
%    输入：距离矩阵 D，低维空间维数 d
%    输出：Z 每一列都是样本的低维坐标
%%
D2 = D .* D;
di = mean(D2, 2);
dj = mean(D2);
dd = mean(D2(:));
% 降维后样本的内积矩阵
B = - (D2 - di - dj + dd) / 2;
[Vd, Dd] = eigs(B, d, 'largestabs');
Z = sqrt(Dd) * Vd';
end