function F = filtpsdaliased(x,xdata)
kT = 4.1;
%et = 1/(2*xdata(end));
%et = 0.65e-3;
%et = 0.30e-3;
et = .650e-3;
F(length(xdata))=0;

for i=1:length(xdata);
    for j=-40:40;
        F(i) = (kT./((pi^2).*x(1).*((xdata(i) + (j-1)*(2*xdata(end))).^2+x(2)^2))).*(sinc((xdata(i) + (j-1)*(2*xdata(end))).*et)).^2 + F(i);
    end
end