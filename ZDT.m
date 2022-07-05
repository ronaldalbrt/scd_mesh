function z = ZDT(x, func)
    n = numel(x);
    if func == 1
        f1=x(1);
        g=1+9/(n-1)*sum(x(2:end));
        h=1-sqrt(f1/g);
        f2=g*h;
        z=[f1 f2];
    elseif func == 4
        f1 = x(1);
        g = 1 + 10*(n-1) + sum(x(2:end).^2 - 10*cos(4*pi*x(2:end)));
    
        h = 1 - sqrt(f1/g);
        f2 = g*h;
        z=[f1 f2];
    end
end