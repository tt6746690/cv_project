function [albedo,wrapped_phase,phase] = SLTriangulation(im,W,depthbounds,spatialfrequency)
%%  Structured light triangulation, solves for shape unknowns (albedo, phase)
%       from demultiplexed images
%
%   im     hxwxS        demultiplexed images under S projector illuminations
%   W      2FxS         optimal bucket multiplexing matrix
%   depthbounds \in [0,1]^(h*w)
%       LB  hxw         pixel-wise phase lower bound
%       UB  hxw         pixel-wise phase upper bound
%   spatialfrequency    number of vertically varying spatial sinusoids in projected patterns
%
    [h,w,S] = size(im);

    shifts = transpose((0:S-1)*2*pi/S);
    L = [ones(S,1) cos(shifts) -sin(shifts)];
    b = (W*L)';
    % multiplexed images
    A = reshape(im,[],S)*W';
    % shape unknowns
    U = A/b;
    % albedo
    albedo = sqrt(sum(U(:,2:3).^2,2));
    % phase \in [0,pi]
    phase = acos(U(:,2)./(albedo+(albedo==0)));
    % phase \in [pi,2pi]
    phase = sign(U(:,3)).*phase;
    phase = mod(real(phase),2*pi);

    [wrapped_phase,albedo] = deal(reshape(phase,h,w),reshape(albedo,h,w));

    % phase unwrapping via depth bound
    phase = refinePhaseUsingDisparityBound(wrapped_phase,spatialfrequency,depthbounds.LB,depthbounds.UB);
end