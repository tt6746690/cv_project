% The code can only be used for research purpose.

% Please cite the following paper when you use it:
   %Lei Zhang, R. Lukac, X. Wu and D. Zhang, 
   %“PCA-based Spatial Adaptive Denoising of CFA Images for Single-Sensor Digital Cameras,” 
   %IEEE Trans. on Image Processing, 2009.

%Note:
%1. The code is not optimized and may have bugs. There are ways to improve the efficiency of the algorithms. 
%   Your revision and improvement are welcome!
%2. In the denoising procedure, the boundary areas will not be processed. You may extend the image boundary (e.g. using
%   symmetrical extension) to process the full image. 

clear;

%%add noise to the image
I=imread('kodak_fence.tif','tif');
I=double(I);
[n,m,ch]=size(I);
%%%%%%%%%%%%%
v=12;
vr=13/12;
vb=10/12;
vg=1;

noi=v*randn(n,m);
In(:,:,3)=I(:,:,3)+vb*noi;
In(:,:,1)=I(:,:,1)+vr*noi;
In(:,:,2)=I(:,:,2)+vg*noi;

%%%%%%%%%%1. downsampling to Bayer pattern: CFA noisy image%%%% 
%%noisy image
mI(1:n,1:m)=In(:,:,2);
mI(1:2:n,2:2:m)=In(1:2:n,2:2:m,1);
mI(2:2:n,1:2:m)=In(2:2:n,1:2:m,3);
%%noiseless image
tI(1:n,1:m)=I(:,:,2);
tI(1:2:n,2:2:m)=I(1:2:n,2:2:m,1);
tI(2:2:n,1:2:m)=I(2:2:n,1:2:m,3);

snro=csnr(mI,tI,20,20)

n1=400;n2=600;m1=200;m2=450;

figure(1),clf;
imshow(I(n1:n2,m1:m2,:)/255);
figure(2),clf;
imshow(mI(n1:n2,m1:m2), [0 255]);
figure(3),clf;
imshow(tI(n1:n2,m1:m2), [0 255]);

%%%%%%%%%%%2. decomposition%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w=10;
f=getgau2(3,w);
fmI=conv2(f,mI);
fmI=fmI(w+1:w+n,w+1:w+m);%%low-pass
%fmI=0;%%you may ignore the decomposition process by setting fmI=0
smI=mI-fmI;%%high-pass

%%%%%%%%%%3. PCA denoising%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dI=smI;
s=6;%%% default variable block size
k=34;%% training block size,(k-s)/2 should be an even integer%%%%%%%%%
k2=k/2;

%Bayer Pattern: [G] R; B G

%%noise pattern
c=1.1;
D=zeros(s,s);
D(:,:)=c*vg*v;
D(1:2:s,2:2:s)=c*vr*v;
D(2:2:s,1:2:s)=c*vb*v;
D=D.^2;

%%denoising loop
for i=1:2:n-k
   for j=1:2:m-k
      Block=smI(i:i+k-1,j:j+k-1);%k by k block
      dB=pca_cfa(Block,D,s);%pca denoising
      dI(i-1+k2:i+k2,j-1+k2:j+k2)=dB(3:4,3:4);
   end
end

dI=dI+fmI;
snrf=csnr(dI,tI,20,20)
figure(4),clf;
imshow(dI(n1:n2,m1:m2), [0 255]);

%%%%%%%%%%%%4. color demosaicking
%We use the following method for color demosaicking
%L. Zhang and X. Wu, “Color demosaicking via directional linear minimum mean square-error estimation,” 
%IEEE Trans. on Image Processing, vol. 14, pp. 2167-2178, Dec. 2005.

dmI=dmsc(dI);
snrcdm=csnr(I,dmI,20,20)
figure(5),clf;
imshow((dmI(n1:n2,m1:m2,:))/255);

