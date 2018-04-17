Sigma = [1, 0; 0, 8];
mu1 = [0, 0];
xi = mvnrnd(mu1, Sigma, 100);
mu2 = [9, -18];
x2 = mvnrnd(mu2, Sigma, 200);
mu3 = [18, 0];
x3 = mvnrnd(mu3, Sigma, 100);
% Show the data points
plot([xi(:,1); x3(:, 1)], [xi(:,2); x3(:, 2)], 'r^'); hold on;
plot(x2(:,1), x2(:,2), 'bo');
axis equal

%% LDA
class1 = [xi; x3]';
class2 = x2';
mean1 = mean(class1, 2);
mean2 = mean(class2, 2);
mean12 = mean([class1, class2], 2);
Sw = (class1 - mean1) * (class1 - mean1)' + (class2 - mean2) * (class2 - mean2)';
Sb = 200 * (mean1 - mean12) * (mean1 - mean12)' + 200 * (mean2 - mean12) * (mean2 - mean12)';
W = Sw \ (mean1 - mean2);
W = 80 .* W / sqrt(W' * W);
XY = [W + mean12, -W + mean12];
line(XY(1, :), XY(2, :), 'Color','b','LineStyle','-')

%% NNDA
SwNNDA = zeros(2, 2);
for xi = class1
    distb = min(sqrt(sum((xi - class2) .* (xi - class2))));
    dw = sum((xi - class1) .* (xi - class1));
    di = unique(dw);
    disti = di(2);
    wi = disti / (disti + distb);
    xj = class1(:, dw == disti);
    SwNNDA = SwNNDA + wi .* (xi - xj) * (xi - xj)';
end

for xi = class2
    distb = min(sqrt(sum((xi - class1) .* (xi - class1))));
    dw = sum((xi - class2) .* (xi - class2));
    di = unique(dw);
    disti = di(2);
    wi = disti / (disti + distb);
    xj = class2(:, dw == disti);
    SwNNDA = SwNNDA + wi .* (xi - xj) * (xi - xj)';
end
% End SwNNDA

SbNNDA = zeros(2, 2);
for xi = class1
    db = sqrt(sum((xi - class2) .* (xi - class2)));
    distb = min(db);
    dw = sum((xi - class1) .* (xi - class1));
    di = unique(dw);
    disti = di(2);
    wi = disti / (disti + distb);
    xj = class2(:, db == distb);
    SbNNDA = SbNNDA + wi .* (xi - xj) * (xi - xj)';
end

for xi = class2
    db = sqrt(sum((xi - class1) .* (xi - class1)));
    distb = min(db);
    dw = sum((xi - class2) .* (xi - class2));
    di = unique(dw);
    disti = di(2);
    wi = disti / (disti + distb);
    xj = class1(:, db == distb);
    SbNNDA = SbNNDA + wi .* (xi - xj) * (xi - xj)';
end
% End SbNNDA

WNNDA = SwNNDA \ (mean1 - mean2);
WNNDA = 80 .* WNNDA / sqrt(WNNDA' * WNNDA);
XY = [WNNDA + mean12, -WNNDA + mean12];
line(XY(1, :), XY(2, :), 'Color','k','LineStyle','--')
legend('Class1', 'Class2', 'LDA', 'NNDA')
axis([-20 40 -40 20])
grid on