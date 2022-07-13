function [Xhat,history] = ADMM(Y,A,InitEstFunc,params,orig_im)
    %   Run ADMM to solve 
    %       xhat(Y) = \argmin_x 0.5*||Ax-y||_2^2 + λ*0.5*x'*(x-denoise(x)) where y = vec(Y)
    %   with the augmented lagrangian of form 
    %       L_{ρ}(x,z) = 0.5*||Ax-y||_2^2 + λ*0.5*<z,z-denoise(z)> + <u,(x-z)> + (ρ/2)*||x-z||_2^2
    %   
    % Inputs:
    %   y                        input image
    %   A                        multiplexing and spatial subsampling operator
    %   InitEstFunc              initial guess of `y` based on `x`
    %   params                   parameters related to optimization
    % 
    % Outputs:
    %   Xhat                     imputed image
    %   history                  psnr/ssim
 
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

    x = InitEstFunc(Y);
    z = x;
    u = zeros(size(x));

    [h,w,S] = size(x);
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

        x_old = x;
        z_old = z;
        u_old = u;

        % primal x update

        if x_update_fast
            x = z-u;
            x = x(:) + A'*( (Y(:) - A*x(:))./(zeta+rho) );
            x = ToIm( Clip(x,0,255) );
        else
            x = ToIm(A'*Y(:))+rho*(z-u);
            x = R\(R'\(x(:)));
            x = ToIm( Clip(x,0,255) );
        end


        % primal z update

        switch z_update_method
        case "fixed_point"
            for j = 1:1:inner_denoiser_iters
                denoised_z = Denoiser(z,effective_sigma,denoiser_type);
                z = (rho*(x+u) + lambda*denoised_z)/(lambda + rho);
            end
        case "denoiser"
            z = Denoiser(x+u,lambda/rho,denoiser_type);
        otherwise
            warning("v-update method not correct");
        end
    
        % scaled dual u update
        
        u = u + x - z;
        
        % print/save
        
        if compute_psnr_ssim
            [psnr,ssim] = ComputePSNRSSIM(orig_im, x);
            history.psnrs = [history.psnrs psnr];
            history.ssims = [history.ssims ssim];
            if verbose && (mod(k,floor(outer_iters/10)) == 0 || k == outer_iters)
                fprintf('ADMM-%s (k=%3d) sigma:%.1f\t PSNR/SSIM: %2.2f/%.4f\n',...
                    upper(denoiser_type),k,effective_sigma,psnr,ssim);
                % imshow(mat2gray(FlattenChannels(orig_im,x,z,u)));
            end
            
            if ~strcmp(save_iterates,'')
                imwrite(uint8(FlattenChannels(x)),...
                    sprintf('%s/Iter=%03d_PSNR=%2.2f_SSIM=%.4f.png',...
                        save_iterates,k,psnr,ssim));
            end
        end
    end
    
    Xhat = Clip(x,0,255);
end