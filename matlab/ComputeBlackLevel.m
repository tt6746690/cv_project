function ComputeBlackLevel(blackimsdir,h,w,saveto)
%   Compute black levels by stacking dark frames in `blackimsdir` 
%       saves result to `saveto`
%
%   >> saveto = "~/github/cv_project/data/blacklevel_all1/blacklevel.mat";
%   >> blackimsdir = "~/github/cv_project/data/blacklevel_all1";
%   >> ComputeBlackLevel(blackimsdir,176,288,savedir);
%
    blacklvl = zeros(2,h,w);
    blackstd = zeros(2,h,w);

    for bktno = 1:2
        imfiles = dir(sprintf("%s/bucket%d*.png",blackimsdir,bktno));
        ims = zeros(h,w,numel(imfiles));
        for i = 1:numel(imfiles)
            ims(:,:,i) = double(imread(sprintf("%s/%s",blackimsdir,imfiles(i).name)));
        end
        blacklvl(bktno,:,:) = mean(ims,3);
        blackstd(bktno,:,:) = std(ims,0,3);
    end
    if saveto ~= ""
        save(saveto,'blacklvl','blackstd');
    end
end