function f=getgau2(s,l)

f=zeros(2*l+1,2*l+1);
i=0;

for x=-l:l
    i=i+1;j=0;
    for y=-l:l
        j=j+1;
        f(i,j)=exp(-(x^2+y^2)/(2*s^2))/(2*pi*s*s);
    end
end

a=sum(sum(f));
f=f/a;
return;