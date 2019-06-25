function b = isInteger(x)
    b = isfinite(x) & x==floor(x);
end
