clear variables;
close all;
clc;


%% 

width = 50;
T1 = 17;
T2 = 31;
T = T1*T2;

xs = 1:0.1:T;

% the spatial sinusoids
I1 = cos((2*pi)*(1/T1)*xs);
I2 = cos((2*pi)*(1/T2)*xs);


% relative phase (in pixels)
rx1 = mod(xs,T1);
rx2 = mod(xs,T2);
% add noise then denoise
mu = 0;
sigma=0.5;
rx1n = rx1 + normrnd(mu,sigma,1,size(xs,2));
rx2n = rx2 + normrnd(mu,sigma,1,size(xs,2));
rx1f = wdenoise(rx1n);
rx2f = wdenoise(rx2n);

P = [rx1;rx2]'-1;    % hproj x S
% Gf = [rx1f;rx2f]'-1;
X = [rx1;rx2]'-1;    % h x S
X = X(1:10:end,:);


% builds table: from patterns
%     Table[correspondence in pixel space] = absolute phase for that pixel
absphases = [];
for i = 1:size(P,1)
    absphase = Chinese(P(i,:),[T1,T2]);
    absphases = [absphases absphase];
end


% nearest neighbor search 
D = pdist2(P,X,'euclidean');
[M,I] = min(D,[],1);
XX = absphases(I);


% DecodeZNCC 
% normalize_ = @(x) normalize(normalize(x,2,'center','mean'),2,'norm',2);
% X = normalize_(double(X));
% P = normalize_(double(P));
% zncc = X*P';
% XX = max(zncc,[],2)';

% asnccaccGC.m
% imgs2np = build_as_2d_np(double(X));
% asnccgtnp = build_asncc_2d_np(double(P));
% scores = imgs2np * asnccgtnp';
% [~, XX] = max(scores,[],2);
% XX = XX';





% XX = absphases(I);

% norm(XX-X)


% mesh(flipud([repmat(cos(2*pi*(1/T1)*rx1),width,1);
%              repmat(cos(2*pi*(1/T2)*rx2),width,1);
%              repmat(cos(2*pi*(1/T1)*rx1f),width,1);
%              repmat(cos(2*pi*(1/T2)*rx2f),width,1);
%              repmat(cos(2*pi*(1/T)*X),width,1);
%              repmat(cos(2*pi*(1/T)*XX),width,1)]))
% colormap('gray');
% view(2);
% grid off;
% set(gca,'XTick',[], 'YTick', [], 'ZTick', []);
% xlim([-10 size(xs,2)]);
%
         
% imagesc(flipud([repmat(cos(2*pi*(1/T1)*rx1n(1:10:end)),width,1);
%              repmat(cos(2*pi*(1/T2)*rx1f(1:10:end)),width,1);
%              repmat(cos(2*pi*(1/T)*XX),width,1)]))
%          
imagesc( [rx1(1:10:end); rx2(1:10:end); XX] )
         
         
colormap('gray');
view(2);
% grid off;
% set(gca,'XTick',[], 'YTick', [], 'ZTick', []);
xlim([-10 size(xs,2)]);
% print(gcf,'../writeup/assets/phase_shifting_chinese_reminder_lower_freq_noisy_wdenoiser_nearest_neighbor.png','-dpng','-r500')

%% why AA^T diagonal ?


S = 6;
h = 4;
w = 4;
P = h*w;
W = BucketMultiplexingMatrix(S);
% W = rand(2*(S-1),S);
mask = SubsamplingMask('random',h,w,S-1);
% 
%[A,B,C] = SubsampleMultiplexOperator(S,mask);
P = size(mask,1)*size(mask,2);

B = SubsamplingOperator(mask);
B = blkdiag(B,B);

C = kron(W,speye(P));
A = B * C;
%
A = full(A);
B = full(B);

AAT = A*A'; % 2P x 2P
WWTIp = kron(W*W',eye(P));
I = find(WWT(1,:));

BBB = B*kron(W*W',eye(P))*B';

for i = 1:2*P
    i, find(WWTIp(i,:))
end

find(W*W'==3)
Y = zeros(h,w,2); y = Y(:); y(1)=1;
BTy = B'*y;
WWTBTy = WWTIp*BTy

find(BTy)
find(WWTBTy)

[Q,D,QT] = eig(W*W');  % W*W' = QDQ^T

AAT1 = B*kron(W*W',eye(P))*B';
AAT2 = B*kron(Q*D*Q',eye(P))*B';
AAT3 = B*kron(Q,eye(P))*kron(D,eye(P))*kron(Q',eye(P))*B';
AAT4 = B*kron(D,eye(P))*B';

[L,U,P] = lu(W*W');  % P^-1 L U = A




