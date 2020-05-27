clear variables;
close all;
clc;


%% 

width = 50;
T1 = 11;
T2 = 17;
T3 = 31;
T = T1*T2*T3;

xs = 1:T;

% the spatial sinusoids
I1 = cos((2*pi)*(1/T1)*xs);
I2 = cos((2*pi)*(1/T2)*xs);
I2 = cos((2*pi)*(1/T3)*xs);


% relative phase (in pixels)
rx1 = mod(xs,T1);
rx2 = mod(xs,T2);
rx3 = mod(xs,T3);
% add noise then denoise
mu = 0;
sigma = range(rx1) * 0.1;
rx1n = rx1 + normrnd(mu,sigma,1,size(xs,2));
rx2n = rx2 + normrnd(mu,sigma,1,size(xs,2));
rx3n = rx3 + normrnd(mu,sigma,1,size(xs,2));
rx1f = wdenoise(rx1n);
rx2f = wdenoise(rx2n);
rx3f = wdenoise(rx3n);

G = [rx1;rx2;rx3]'-1; 
% G = G(1:20:end,:);
Gf = [rx1f;rx2f;rx3f]'-1;
base = [T1,T2,T3];

X = [];
for i = 1:size(G,1)
    g = G(i,:);
    x = Chinese(g,base)
    X = [X x];
end


% nearest neighbor search 

D = pdist2(G,Gf,'euclidean');
[M,I] = min(D,[],1);
XX = X(I);

norm(XX-X)


%% 

plot(XX); hold on; plot(X); legend


%% 

mesh(flipud([repmat(cos(2*pi*(1/T1)*rx1),width,1);
             repmat(cos(2*pi*(1/T2)*rx2),width,1);
             repmat(cos(2*pi*(1/T3)*rx3),width,1);
             repmat(cos(2*pi*(1/T1)*rx1f),width,1);
             repmat(cos(2*pi*(1/T2)*rx2f),width,1);
             repmat(cos(2*pi*(1/T3)*rx3f),width,1);
             repmat(cos(2*pi*(1/T)*X),width,1);
             repmat(cos(2*pi*(1/T)*XX),width,1)]))
colormap('gray');
view(2);
grid off;
set(gca,'XTick',[], 'YTick', [], 'ZTick', []);
xlim([-10 size(xs,2)]);
%%
         
mesh(flipud([repmat(cos(2*pi*(1/T1)*rx1n),width,1);
             repmat(cos(2*pi*(1/T2)*rx1f),width,1);
             repmat(cos(2*pi*(1/T)*XX),width,1)]))
         
colormap('gray');
view(2);
grid off;
set(gca,'XTick',[], 'YTick', [], 'ZTick', []);
xlim([-10 size(xs,2)]);
% print(gcf,'../writeup/assets/phase_shifting_chinese_reminder_lower_freq_noisy_wdenoiser_nearest_neighbor.png','-dpng','-r500')
