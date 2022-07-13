% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Computes the PSNR of two single-channel images

function out_psnr = ComputePSNR(orig_im,est_im,varargin)

    normalize=false;
    % Map of parameter names to variable names
    params_to_variables = containers.Map( ...
        {'Normalize'}, ...
        {'normalize'});
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

    if normalize == true
        orig_im = mat2gray(orig_im);
        est_im = mat2gray(est_im);
    end

    err = sqrt(mean((orig_im(:)-est_im(:)).^2));
    out_psnr = 20*log10(255/err);

return;