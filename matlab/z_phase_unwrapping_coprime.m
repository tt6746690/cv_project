clear variables;
close all;
clc;


%% 

width = 50;
T1 = 17;
T2 = 31;
T = T1*T2;

xs = 1:0.05:T;

% the spatial sinusoids
I1 = cos((2*pi)*(1/T1)*xs);
I2 = cos((2*pi)*(1/T2)*xs);

% relative phase (in pixels)
rx1 = mod(xs,T1);
rx2 = mod(xs,T2);
% add noise
mu = 0;
sigma = range(rx1) * 0.1;
rx1 = rx1 + normrnd(mu,sigma,1,size(xs,2));
rx2 = rx2 + normrnd(mu,sigma,1,size(xs,2));


% medfilter
%rx1f = medfilt1(rx1,10);
%rx2f = medfilt1(rx2,10);
rx1f = wdenoise(rx1);
rx2f = wdenoise(rx2);

G = [rx1;rx2]'-1;
Gf = [rx1f;rx2f]'-1;
base = [T1,T2];

X = [];
for i = 1:size(xs,2)
    g = Gf(i,:);
    x = Chinese(g,base)
    X = [X x];
end

%% 

plot(rx1); hold on; plot(rx1f)


%% 

% mesh(flipud([repmat(cos(2*pi*(1/T1)*rx1),width,1);
%              repmat(cos(2*pi*(1/T2)*rx2),width,1);
%              repmat(cos(2*pi*(1/T1)*rx1f),width,1);
%              repmat(cos(2*pi*(1/T2)*rx2f),width,1);
%              repmat(cos(2*pi*(1/T)*X),width,1)]))
%          
         
mesh(flipud([repmat(cos(2*pi*(1/T1)*rx1),width,1);
             repmat(cos(2*pi*(1/T1)*rx1f),width,1);
             repmat(cos(2*pi*(1/T)*X),width,1)]))
         
colormap('gray');
view(2);
grid off;
set(gca,'XTick',[], 'YTick', [], 'ZTick', []);
xlim([-10 size(xs,2)]);
print(gcf,'../writeup/assets/phase_shifting_chinese_reminder_lower_freq_noisy_wdenoiser.png','-dpng','-r500')
