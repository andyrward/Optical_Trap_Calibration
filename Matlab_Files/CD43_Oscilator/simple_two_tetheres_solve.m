F  = 25;

P  = 0.41;
lpbp = 0.45;
kT = 4.11;
Lb = 140*lpbp;


% La      = (78+14) * 0.45;
% syms fa(x)
% fa(x) = (1/4*(1 - x/La)^-2 - 1/4 + x/La) - F*P/kT;
% sol = vpasolve(fa)

%%

Lc_range = 5:1:140;

for i = 1:length(Lc_range)
    
    Lc = Lc_range(i)*lpbp;
    syms fb(x)
    fb(x) =  kT/P*(1/4*(1 - x/Lc)^-2 - 1/4 + x/Lc) ...
        + kT/P*(1/4*(1 - x/Lb)^-2 - 1/4 + x/Lb) ...
        - F;
    
    sol = vpasolve(fb);
    output(i) = sol(1);
    
end

for i = 1:length(Lc_range)
    Lc = Lc_range(i)*lpbp;   
    syms fb(x)
    fb(x) =  kT/P*(1/4*(1 - x/Lc)^-2 - 1/4 + x/Lc) ...
        + 0*kT/P*(1/4*(1 - x/Lb)^-2 - 1/4 + x/Lb) ...
        - F;
    
    sol = vpasolve(fb);
    output2(i) = sol(1);
    
    
end


%%
figure;

subplot(2,1,1);
plot(Lc_range, output);hold on;
plot(Lc_range, output2);hold on;
plot(Lc_range, Lc_range*0.45)
xlim([5, 140])
ylabel('Distance (nm)')

subplot(2,1,2);
plot(Lc_range,output2 -  output);
xlabel('# of T with 140 bp loop size')
xlim([5, 140])
ylabel('Distance (nm)')

