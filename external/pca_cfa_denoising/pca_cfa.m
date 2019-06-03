function B=pca_cfa(I,D,w)

%%pca-based denoising of CFA image
%%I:   input n*m noisy block
%%D:   noise pattern

%%%1. form the training data%%%%%%%%%%%%%%%%%
[n,m]=size(I);%% n and m must be even numbers
N=(n-w)/2+1;M=(m-w)/2+1;
L=N*M;%% the number of trials
X=zeros(w*w,L);
k=0;
DN=zeros(1,w*w);
for i=1:w
   for j=1:w
      k=k+1;
      X(k,:)=reshape(I(i:2:n-w+i,j:2:m-w+j),1,L);
      DN(k)=D(i,j);
   end
end

%%%2. Grouping %%%%%%%%%%%%%%%%%%%%%%%%%
q=(L+1)/2;
Xc=X(:,q);
XC=repmat(Xc,1,L);

E=abs(X-XC);
mE=mean(E);
[val,ind]=sort(mE);

num=100;%you may varying this number for better results
X=X(:,ind(1:num));

%%%3. PCA transformation %%%%%%%%%%%%%%%%
[Y, P, V, mX] =getpca2(X,DN);
Y1=0*Y;
%r=8;
for i=1:w*w-8
   y=Y(i,:);
   p=P(i,:);
   p=p.^2;
   nv=sum(p.*DN);
   py=mean(y.^2)+0.01;
   t=max(0,py-nv);
   c=t/py;
   Y1(i,:)=c*Y(i,:);
end

%%%4. inverse PCA transform
B=(P'*Y1+mX);

%%%5. reshape the denoised data
B=B(:,1);
B=reshape(B,w,w);B=B';
return;