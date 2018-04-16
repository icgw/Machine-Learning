Sigma = [1, 0; 0, 1];
mu1 = [1, -1];
x1 = mvnrnd(mu1, Sigma, 200);
mu2 = [5, -4];
x2 = mvnrnd(mu2, Sigma, 200);
mu3 = [1, 4];
x3 = mvnrnd(mu3, Sigma, 200);
mu4 = [6, 4.5];
x4 = mvnrnd(mu4, Sigma, 200);
mu5 = [7.5, 0.0];
x5 = mvnrnd(mu5, Sigma, 200);

% Show the data points
plot(x1(:,1), x1(:,2), 'r.'); hold on;
plot(x2(:,1), x2(:,2), 'b.');
plot(x3(:,1), x3(:,2), 'k.');
plot(x4(:,1), x4(:,2), 'g.');
plot(x5(:,1), x5(:,2), 'm.'); hold off;

%% ≤‚ ‘»Áœ¬
rng('default');
lambda = 1e-3;

sample = zeros(3, 120, 5);
sample(:, :, 1) = [x1(1:120, :), ones(120, 1)]';
sample(:, :, 2) = [x2(1:120, :), ones(120, 1)]';
sample(:, :, 3) = [x3(1:120, :), ones(120, 1)]';
sample(:, :, 4) = [x4(1:120, :), ones(120, 1)]';
sample(:, :, 5) = [x5(1:120, :), ones(120, 1)]';

test = [x1(121:end, :)', x2(121:end, :)', x3(121:end, :)', ...
    x4(121:end, :)', x5(121:end, :)'];
test = [test; ones(1, 400)];

pred = softmaxRegression(sample, lambda);
label = pred(test);
corr = ones(80, 1) .* (1:5);
corr = corr(:);
pre = sum(r == corr) / 400