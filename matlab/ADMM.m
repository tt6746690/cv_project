function [im_out,psnr_out,ssim_out,statistics,iter_ims] = ADMM(y,H,InitEstFunc,params,orig_im)
    %   Run ADMM to solve minimize 
    %       E(x) = ||Hx-y||_2^2 + λ * 0.5*x'*(x-denoise(x))
    %   with the augmented lagrangian of form 
    %       L_{ρ}(x,v) = ||Hx-y||_2^2 + λ * 0.5*x'*(x-denoise(x)) + 
    %                        μ^T(x-v) + (ρ/2) ||x-z||_2^2
    %   
    % Inputs:
    %   y                        input image
    %   H                        multiplexing and spatial subsampling operator
    %   InitEstFunc              initial guess of `y` based on `x`
    %   params.
    %       lambda               relative scaling of data term and regularization term
    %       rho                  augmented lagrangian parameter
    %       outer_iters          number of ADMM iterations
    %       inner_denoiser_iters number of fixed iterations in v-minimization step
    %       denoiser_type        denoiser used in v-minimization step
    %       effective_sigma      input noise level to denoiser
    % 
    % Outputs:
    %   im_out                   imputed image
    %   psnr_out                 psnr of `im_out` to `orig_im`
    %   ssim_out                 ssim of `im_out` to `orig_im`
    %   statistics               psnrs at each iterations
 
    QUIET = 0;
    PRINT_MOD = floor(params.outer_iters/10);
    if ~QUIET
        fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR/SSIM', 'objective');
    end

    lambda = params.lambda;
    rho = params.rho;
    outer_iters = params.outer_iters;
    inner_denoiser_iters = params.inner_denoiser_iters;
    denoiser_type = params.denoiser_type;
    effective_sigma = params.effective_sigma;
    v_update_method = params.v_update_method;
    
    x_est = InitEstFunc(y);
    v_est = x_est;
    u_est = zeros(size(x_est));
    history = zeros(3,0);
    save_iter = 1;

    [h,w,S] = size(x_est);
    ToIm = @(x) reshape(x,h,w,[]);
    
    [R,flag] = chol(H'*H + rho*speye(size(H,2),size(H,2)));
    if flag ~= 0
        warning("H'H+rho*I should be symmetric positive definite");
    end

    iter_ims = zeros(5*h,S*w,outer_iters);
    
    for k = 1:outer_iters
        % iter_ims(:,:,k) = 3*FlattenChannels(x_est,v_est,u_est,...
        % Clip(ToIm(R\(R'\(reshape(ToIm(H'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est, ...
        % Denoiser(Clip(ToIm(R\(R'\(reshape(ToIm(H'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est,effective_sigma,denoiser_type));
        % imshow(iter_ims(:,:,k)/255);
        % pause;

        x_old = x_est;
        v_old = v_est;
        u_old = u_est;
    
        % primal x update
        x_est = ToIm(H'*y(:))+rho*(v_est-u_est);
        x_est = R\(R'\(x_est(:)));
        x_est = ToIm(x_est);
        x_est = Clip(x_est,0,255);
        
        % primal v update
        switch v_update_method
        case "fixed_point"
            for j = 1:1:inner_denoiser_iters
                f_v_est = Denoiser(v_est,effective_sigma,denoiser_type);
                v_est = (rho*(x_est+u_est) + lambda*f_v_est)/(lambda + rho);
            end
        case "denoiser"
            v_est = Denoiser(x_est+u_est,lambda/rho,denoiser_type);
        otherwise
            warning("v-update method not correct");
        end
    
        % scaled dual u update
        u_est = u_est + x_est - v_est;
    
        if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
            f_est = Denoiser(x_est,effective_sigma,denoiser_type);
            costfunc = norm(reshape(ToIm(H*x_est(:))-y,[],1)) + lambda*x_est(:)'*(x_est(:)-f_est(:));
            im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2),:);
            [psnr,ssim] = ComputePSNRSSIM(orig_im, im_out);

            fprintf('%7i %.5f/%.5f %12.5f \n',k,psnr,ssim,costfunc);
            history(:,save_iter) = [psnr;ssim;costfunc];
            save_iter = save_iter + 1;

            % imshow(3*FlattenChannels(orig_im,x_est,v_est,u_est,...
            %     Clip(ToIm(R\(R'\(reshape(ToIm(H'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est, ...
            %     Denoiser(Clip(ToIm(R\(R'\(reshape(ToIm(H'*y(:))+rho*(v_est-u_est),[],1)))),0,255)+u_est,lambda/rho,denoiser_type))/255);

            imshow(2*FlattenChannels(orig_im,x_old,x_est,v_old,v_est,u_old,u_est)/255);
        end
    end
    
    im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2),:);
    im_out = Clip(x_est,0,255);
    [psnr_out,ssim_out] = ComputePSNRSSIM(orig_im, im_out);

    statistics.psnr     = history(1,:);
    statistics.ssim     = history(2,:);
    statistics.costfunc = history(3,:);
end