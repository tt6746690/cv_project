function relphase = PhaseShiftingSolveRelativePhase(I, phase_shift)
%   Closed form solution for relative phase for phase shifting structured light
%       I             h x w x K
%           Image Itensity where K is number of shifted phases
%       phase_shift   \in [0,2pi]^K
%           phase shifting of projectedf patterns
%
    [h,w,K] = size(I);
    num = I.*repmat(reshape(sin(phase_shift),1,1,K),[h w 1]);
    den = I.*repmat(reshape(cos(phase_shift),1,1,K),[h w 1]);
    relphase = atan(- sum(num,3) ./ sum(den,3));
end