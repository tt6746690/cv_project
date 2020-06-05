function [Xhat,history] = ADMM(y,A,InitEstFunc,params,orig_im)
    %   Run ADMM to solve minimize 
    %       E(x) = ||Ax-y||_2^2 + λ * 0.5*x'*(x-denoise(x))
    %   with the augmented lagrangian of form 
    %       L_{ρ}(x,v) = ||Ax-y||_2^2 + λ * 0.5*x'*(x-denoise(x)) + 
    %                        μ^T(x-v) + (ρ/2) ||x-z||_2^2
    %   
    % Inputs:
    %   y                        input image
    %   A                        multiplexing and spatial subsampling operator
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
    %   history                  psnr/ssim/costfunc
 
    verbose                 = params.verbose;
    lambda                  = params.lambda;
    rho                     = params.rho;
    outer_iters             = params.outer_iters;
    inner_denoiser_iters    = params.inner_denoiser_iters;
    denoiser_type           = params.denoiser_type;
    effective_sigma         = params.effective_sigma;
    z_update_method         = params.z_update_method;
    save_iterates           = params.save_iterates;
    compute_psnr_ssim       = params.compute_psnr_ssim;
    
    if ~strcmp(save_iterates,'') && ~exist(save_iterates,'dir')
        mkdir(save_iterates);
    end

    x_est = InitEstFunc(y);
    z_est = x_est;
    u_est = zeros(size(x_est));

    [h,w,S] = size(x_est);
    ToIm = @(x) reshape(x,h,w,[]);
    
    % precomputation for x-update
    x_update_fast = isdiag(A*A');
    if x_update_fast
        zeta = full(diag(A*A'));
    else
        [R,flag] = chol(A'*A + rho*speye(size(A,2)));
        if flag ~= 0
            warning("A'A + rho*I should be symmetric positive definite");
        end 
    end 
    
    history.psnrs = [];
    history.ssims = [];
    
    for k = 1:outer_iters

        x_old = x_est;
        z_old = z_est;
        u_old = u_est;

        % primal x update

        if x_update_fast
            x_est = z_est-u_est;
            x_est = x_est(:) + A'*( (y(:) - A*x_est(:))./(zeta+rho) );
            x_est = ToIm( Clip(x_est,0,255) );
        else
            x_est = ToIm(A'*y(:))+rho*(z_est-u_est);
            x_est = R\(R'\(x_est(:)));
            x_est = ToIm( Clip(x_est,0,255) );
        end


        % primal z update

        switch z_update_method
        case "fixed_point"
            for j = 1:1:inner_denoiser_iters
                denoised_z_est = Denoiser(z_est,effective_sigma,denoiser_type);
                z_est = (rho*(x_est+u_est) + lambda*denoised_z_est)/(lambda + rho);
            end
        case "denoiser"
            z_est = Denoiser(x_est+u_est,lambda/rho,denoiser_type);
        otherwise
            warning("v-update method not correct");
        end
    
        % scaled dual u update
        
        u_est = u_est + x_est - z_est;
        
        % print/save
        
        if compute_psnr_ssim
            [psnr,ssim] = ComputePSNRSSIM(orig_im, x_est);
            history.psnrs = [history.psnrs psnr];
            history.ssims = [history.ssims ssim];
            if verbose && (mod(k,floor(outer_iters/10)) == 0 || k == outer_iters)
                fprintf('ADMM-%s (k=%3d) sigma:%.1f\t PSNR/SSIM: %2.2f/%.4f\n',...
                    upper(denoiser_type),k,effective_sigma,psnr,ssim);
                % imshow(mat2gray(FlattenChannels(orig_im,x_est,z_est,u_est)));
            end
            
            if ~strcmp(save_iterates,'')
                imwrite(uint8(FlattenChannels(x_est)),...
                    sprintf('%s/Iter=%03d_PSNR=%2.2f_SSIM=%.4f.png',...
                        save_iterates,k,psnr,ssim));
            end
        end
    end
    
    Xhat = Clip(x_est,0,255);
end