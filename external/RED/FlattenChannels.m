function ims = FlattenChannels(varargin)
    % Given `nargin` images of size `hxwxC` 
    %       return a concatenated image of size `nargin*h x w*C`
    %
    [h,w,C] = size(varargin{1});
    ims = zeros(nargin*h,w*C);
    for i = 1:C
        for j = 1:nargin
            ims(((j-1)*h+1):(j*h),((i-1)*w+1):(i*w)) = varargin{j}(:,:,i);
        end
    end
end