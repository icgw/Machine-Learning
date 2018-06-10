function Z = MDS(D, d)
%Multiple Dimensional Scaling ��ά������
%    ���룺������� D����ά�ռ�ά�� d
%    �����Z ÿһ�ж��������ĵ�ά����
%%
D2 = D .* D;
di = mean(D2, 2);
dj = mean(D2);
dd = mean(D2(:));
% ��ά���������ڻ�����
B = - (D2 - di - dj + dd) / 2;
[Vd, Dd] = eigs(B, d, 'largestabs');
Z = sqrt(Dd) * Vd';
end