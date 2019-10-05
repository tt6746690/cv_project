function [albedo,wrapped_phase,phase] = SLTriangulation(im,W,depthbounds,spatialfrequency,varargin)
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
%   Note the im should be ordered according to phase shifts in projecter patterns,
%       This ordering is not maintanied during image acquisition, so needs to be determined 
%       experimentally, by brute forcing all circular shifts and find shifts for a specific 
%       `S` for a fixed scene that has the same disparity map intensity
%
%

    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'Shifts'}, ...
        {'shifts'});
    v = 1;
    while v <= numel(varargin)
        param_name = varargin{v};
        if isKey(params_to_variables,param_name)
            assert(v+1<=numel(varargin));
            v = v+1;
            % Trick: use feval on anonymous function to use assignin to this workspace
            feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
        else
            error('Unsupported parameter: %s',varargin{v});
        end
        v=v+1;
    end

    [h,w,S] = size(im);

    if ~exist('shifts','var')
        shifts = transpose((0:S-1)*2*pi/S);
    end

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
