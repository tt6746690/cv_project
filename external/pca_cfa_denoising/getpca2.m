function [Y, P, V, mx]=getpca2(X,D)

%X: MxN matrix (M dimensions, N trials)
%Y: Y=P*X
%P: the transform matrix
%V: the variance vector

[M,N]=size(X);

mx=mean(X,2);
mx=repmat(mx,1,N);

X=X-mx;

CovX=X*X'/(N-1);
D=diag(D);
CovX=CovX-D;
ind = find(CovX<0); 
CovX(ind) =0.0001; 

[P,V]=eig(CovX);

V=diag(V);
[t,ind]=sort(-V);
V=V(ind);
P=P(:,ind);
P=P';
Y=P*X;

return;
