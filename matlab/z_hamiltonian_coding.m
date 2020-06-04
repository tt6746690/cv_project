%% visualize coding curves for different patterns
%
clc; clear; close all;
ProjectPaths;

%% coding curve w/ quantization error 
figure('Renderer', 'painters', 'Position', [10 10 600 300])

N = 600; % #encoding
spatial_freqs = [1 3 5 17 31]; % spatial frequency of sinusoids
spatial_freq = 5;
n_shifts = 3;
P = 0.5 + 0.5*cos(spatial_freq*(0:N-1)'*2*pi/N + ((1:n_shifts)-1)*2*pi/n_shifts);

subplot(1,2,1)
plot3(P(:,1),P(:,2),P(:,3),'-o','Color','#89C4F4','MarkerSize',5,'MarkerFaceColor','#59ABE3');
title('No Quantization Error');
xlabel('$P_1(c)$','Interpreter','latex');
ylabel('$P_2(c)$','Interpreter','latex');
zlabel('$P_3(c)$','Interpreter','latex');
grid on;
v = [-2 -1 1]; view(v);

P = floor(P * 24) / 24;
subplot(1,2,2)
plot3(P(:,1),P(:,2),P(:,3),'-o','Color','#89C4F4','MarkerSize',5,'MarkerFaceColor','#59ABE3');
title('With Quantization Error');
xlabel('$P_1(c)$','Interpreter','latex');
ylabel('$P_2(c)$','Interpreter','latex');
zlabel('$P_3(c)$','Interpreter','latex');
grid on;
v = [-2 -1 1]; view(v);

saveas(gcf,'../writeup/assets/coding_curve_quantization_effect.png');

%% coding curve for a la carte code

figure('Renderer', 'painters', 'Position', [10 10 300*5 300])

Ks = [4 8 16 32 64];
for i = 1:5
    K = Ks(i);
    P = readmatrix(sprintf('../external/SLtraining-Python/OptimizedCodes/608/pt01-608-3-%d.txt',K));
    P = P(:,1:3);

    subplot(1,5,i)
    plot3(P(:,1),P(:,2),P(:,3),'-o','Color','#89C4F4','MarkerSize',5,'MarkerFaceColor','#59ABE3');
    xlabel('$P_1(c)$','Interpreter','latex');
    ylabel('$P_2(c)$','Interpreter','latex');
    zlabel('$P_3(c)$','Interpreter','latex');
    title(sprintf('$F\\leq%d$',K),'Interpreter','latex');
    grid on;
    v = [-2 -1 1]; view(v);
end

saveas(gcf,'../writeup/assets/coding_curve_la_carte.png');

%% coding curve for pair-wise coprime sinusoidal curve

Ts = [5 11 13];

N = prod(Ts);
xs = (1:N)';
rx1 = mod(xs,Ts(1));
rx2 = mod(xs,Ts(2));
rx3 = mod(xs,Ts(3));
P(:,1) = 0.5 + 0.5*cos(Ts(1)*(0:N-1)'*2*pi/N);
P(:,2) = 0.5 + 0.5*cos(Ts(2)*(0:N-1)'*2*pi/N);
P(:,3) = 0.5 + 0.5*cos(Ts(3)*(0:N-1)'*2*pi/N);

Is = 1:N
plot3(P(Is,1),P(Is,2),P(Is,3),'-o','Color','#89C4F4','MarkerSize',5,'MarkerFaceColor','#59ABE3');
title('Sinusoids w/ Pairwise Coprime Periods');
xlabel('$P_1(c)$','Interpreter','latex');
ylabel('$P_2(c)$','Interpreter','latex');
zlabel('$P_3(c)$','Interpreter','latex');
grid on;
v = [-2 -1 1]; view(v);

saveas(gcf,'../writeup/assets/coding_curve_sinusoids_pairwise_coprime.png');

