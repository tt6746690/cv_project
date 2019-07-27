function SensorMask = GenerateSensorMask(mask_type,F,S,h,w,filename)
    % Generate sensor mask of size S*h x w for a given mask type and image size
    %       for upload to the camera setup
    %
    M = SubsamplingMask(mask_type,h,w,F);
    W = BucketMultiplexingMatrix(S);
    Cp = W(1:F,:);
    C = zeros(h,w,S);
    for i = 1:h
        for j = 1:w
            C(i,j,:) = Cp(M(i,j),:);
        end
    end
    splitC = num2cell(C, [1 2]);
    SensorMask = vertcat(splitC{:});
    imwrite(logical(SensorMask),sprintf("%s.bmp",filename));
    imwrite(uint8(M),sprintf("%s.png",filename));
end