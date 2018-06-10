N=1000;
%�Ӹ�˹����
noise = 0.001 * randn(1, N);
%��׼�� swiss roll ����
tt = (3 * pi/2) * (1 + 2 * rand(1, N)); height = 21 * rand(1, N);
X = [(tt + noise) .* cos(tt); height; (tt + noise) .* sin(tt)];
%��ʾ����
point_size = 20;

figure('Name', 'Isometric Mapping')
subplot(4, 4, 1)
cla
scatter3(X(1, :), X(2, :), X(3, :), point_size, tt, 'filled');
view([12 12]); grid off; axis off; hold on;
axis off;
axis equal;
drawnow;
% Isomap ʵ��
for k = 6:20
    Z = Isomap(X, k, 2);
    subplot(4, 4, k - 4);
    scatter(Z(1, :), Z(2, :), point_size,tt,'filled');
    title(['k = ', num2str(k), ' ʱ']);
    grid off;
    axis off;
end

% LLE ʵ��
figure('Name', 'Locally Linear Embedding')
subplot(4, 4, 1)
cla
scatter3(X(1, :), X(2, :), X(3, :), point_size, tt, 'filled');
view([12 12]); grid off; axis off; hold on;
axis off;
axis equal;
drawnow;
for k = 6:20
    Z = LLE(X, k, 2);
    subplot(4, 4, k - 4);
    scatter(Z(1, :), Z(2, :), point_size,tt,'filled');
    title(['k = ', num2str(k), ' ʱ']);
    grid off;
    axis off;
end

% LE ʵ��
figure('Name', 'Laplacian Eigenmaps')
subplot(4, 4, 1)
cla
scatter3(X(1, :), X(2, :), X(3, :), point_size, tt, 'filled');
view([12 12]); grid off; axis off; hold on;
axis off;
axis equal;
drawnow;
for k = 6:20
    Z = LE(X, k, 2);
    subplot(4, 4, k - 4);
    scatter(Z(1, :), Z(2, :), point_size,tt,'filled');
    title(['k = ', num2str(k), ' ʱ']);
    grid off;
    axis off;
end