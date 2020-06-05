function f_est = Denoiser(x_est,effective_sigma,denoiser_type)
    % Denoiser `x_est` with `effective_sigma` 
    %
    %

    switch denoiser_type
    case "mf"
        fsize = [3 3];
        if size(x_est,3) ~= 1
            f_est = zeros(size(x_est));
            for i = 1:size(x_est,3)
                f_est(:,:,i) = medfilt2(x_est(:,:,i),fsize);
            end
        else
            f_est = medfilt2(x_est,fsize);
        end
    case "bm3d"
        if size(x_est,3) ~= 1
            f_est = zeros(size(x_est));
            for i = 1:size(x_est,3)
                [~, f_est_] = BM3D(1,x_est(:,:,i),effective_sigma);
                f_est(:,:,i) = f_est_;
            end
        else
            [~, f_est] = BM3D(1,x_est,effective_sigma);
        end
        f_est = f_est*effective_sigma/5;
    case "tnrd"
        if size(x_est,3) ~= 1
            f_est = zeros(size(x_est));
            for i = 1:size(x_est,3)
                f_est(:,:,i) = ReactionDiffusion(5/effective_sigma*x_est(:,:,i));
            end
        else
            f_est = ReactionDiffusion(5/effective_sigma*x_est);
        end
        f_est = f_est*effective_sigma/5;
    otherwise
        warning("no such denoiser")
    end
end
