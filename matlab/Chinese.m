function x = Chinese(g,b)
    % Objective: To find the smallest x such that: 
    %    g=mod(x,b) or written in another way
    %    x === g (mod b)
    %
    % https://www.mathworks.com/matlabcentral/fileexchange/32649-chinese-remainder-theorem-for-integers-simple
    % 
    %
    %     T1 = 4;
    %     T2 = 5;
    %     b = [T1 T2]; % the bases should be relativly prime
    % 
    %     A = repmat(1:T1,1,T2);
    %     B = repmat(1:T2,1,T1);
    %     G = [A;B]'-1;  % minus 1 to start from 0
    % 
    %     X = [];
    %     for i = 1:T1*T2
    %         g = G(i,:);
    %         x = Chinese(g,b);
    %         X = [X x];
    %     end
    %     X


    [bx by] = meshgrid(b, b);
    bb = gcd(bx,by)-diag(b);
    pp = ~sum(sum(bb>1)); 

    if (pp)
        % display(['The Bases [relativly prime] are: b=[' num2str(b) ']'])
        % display(['The Number [representation] is : g=<' num2str(g) '>' ])

        % take out one by one bases and replace with 1's 
        xo = by-diag(b-1);

        % and get the product of the others
        Mk = prod(xo);

        % now we should get an solution for x and xa where Mk.*xa =(%b) x =(%b) 1
        % note that xa.*g is a solution, i.e xa.*g =(%b) g, because xa =(%b) ones
        [Gk, nk, Nk] = gcd ( b, Mk ) ;
        % [G,C,D] = GCD( A, B ) also returns C and D so that G = A.*C + B.*D.
        % These are useful for solving Diophantine equations and computing
        % Hermite transformations.

        % Then the strange step
        Sum_g_Nk_Mk = sum ( (g .* Nk) .* Mk ) ;

        % get the lowest period unique answer between [0:(product(b)-1)]
        x = mod(Sum_g_Nk_Mk,prod(b));

        % display(['The Number [lowest unique value] is: x=''' num2str(x) '''' ])
    else
        display('The Bases are NOT Relprime.')
    end

end