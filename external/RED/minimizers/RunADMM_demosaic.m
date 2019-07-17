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

% Objective:
%   Minimize E(x) = 1/(2sigma^2)||Hx-y||_2^2 + 0.5*lambda*x'*(x-denoise(x))
%   via the ADMM method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems -- ADMM".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   params.lambda - regularization parameter
%   params.beta - ADMM parameter
%   params.outer_iters - number of total iterations
%   params.inner_iters - number of steps to minimize Part1 of ADMM
%   params.inner_denoiser_iters - number of steps to minimize Part2 of ADMM
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY (only the Luminance Y component)

% Outputs:
%   im_out - the reconstructed image
%   psnr_out - PSNR measured between x_est and orig_im

function [im_out, psnr_out, statistics] = RunADMM_demosaic(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, input_sigma, params, orig_im)

% print info every PRINT_MOD steps 
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('printmod: %d\n',PRINT_MOD);
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

% parameters
lambda = params.lambda;
beta = params.beta;
outer_iters = params.outer_iters;
inner_iters = params.inner_iters;
inner_denoiser_iters = params.inner_denoiser_iters;
effective_sigma = params.effective_sigma;
denoiser_type = params.denoiser_type;

% initialization
x_est = InitEstFunc(y);
x_init = InitEstFunc(y);
v_est = x_est;
u_est = x_est*0;
Ht_y = BackwardFunc(y)/(input_sigma^2);

% keept track of convergence
psnrs = zeros(1,0);
costfunc = zeros(1,0);
save_iter = 1;

for k = 1:1:outer_iters

    % imshow(FlattenChannels(x_est)/255);
    % pause;
    
    % Part1 of the ADMM, approximates the solution of:
    % x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2
    % use gradient descent        
    for j = 1:1:inner_iters
        % (v_est - u_est) is z^*
        b = Ht_y + beta*(v_est - u_est);
        A_x_est = BackwardFunc(ForwardFunc(x_est))/(input_sigma^2) + beta*x_est;

        % res = -e_j = (1/\sigma^2) ( H^T*y - H^T*H*z_{j-1} ) + beta*(z^* - z_{j-1})
        res = b - A_x_est;

        % r_j = (1/\sigma^2) H^T*H*e_j + \beta*e_j
        a_res = BackwardFunc(ForwardFunc(res))/(input_sigma^2) + beta*res;

        % `res` is gradient `e_j`
        % `a_res` is `r_j`
        % mu_opt = mean(\mu)
        mu_opt = mean(res(:).*res(:))/mean(res(:).*a_res(:));

        % z_j = z_{j-1} + \mu e_j
        x_est = x_est + mu_opt*res;
        x_est = max( min(x_est, 255), 0);

        % fprintf("gradient mean: %.5f\n",mean(res,'all'));
        % fprintf("step size:     %.5f\n",mu_opt);
        % imshow(FlattenChannels(orig_im,res,Ht_y,x_est)/255);
        % pause;
    end

    
    % relaxation
    x_hat = params.alpha*x_est + (1-params.alpha)*v_est;
    
    
    % Part2 of the ADMM, approximates the solution of
    % v = argmin_z lambda*z'*(z-denoiser(z)) +  0.5*beta||z - x - u||_2^2
    % using gradient descent
    for j = 1:1:inner_denoiser_iters
        f_v_est = Denoiser(v_est, effective_sigma,denoiser_type);
        v_est = (beta*(x_hat + u_est) + lambda*f_v_est)/(lambda + beta);
    end
    
    % Part3 of the ADMM, update the dual variable
    u_est = u_est + x_hat - v_est;
    
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function
        fun_val = CostFunc(y, x_est, ForwardFunc, input_sigma,...
            lambda, effective_sigma,denoiser_type);
        im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2),:);
        psnr_out = ComputePSNR(orig_im, im_out);
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);

        psnrs(save_iter) = psnr_out;
        costfunc(save_iter) = fun_val;
        save_iter = save_iter+1;

        % display image
        imshow(FlattenChannels(orig_im,x_est)/255);
    end
end

im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2),:);
psnr_out = ComputePSNR(orig_im, im_out);

statistics.psnrs = psnrs;
statistics.costfunc = costfunc;

return

end