%% Open Data

[data,fp] = fx_OT_Qick_Process_Plot_Cal_2();


td = array2table(data,'VariableNames',{'t','z','F','V','k','State'});

%% Segment Data on Measuring Voltage
Vmeasure=6.2;

Measure = find_Vmeasure_segments(td.V', Vmeasure);

%% Compute Deltas
pavg=1.8683; % slope of unbound construtc from Oscilator experiment
Nblock=10;
Navg=100;
fig_l=figure;
jump=50;
dz=[];
sdz=[];
j=1;
for i=1:numel(Measure)

m=Measure{i};
m1=m(1);
m2=m(2);
z=td.z(m1:m2);
t=td.t(m1:m2);
F=td.F(m1:m2);
figure(fig_l);
subplot(1,2,1);
hold off;
plot(z);
xlabel("index");
ylabel("Extension (nm)");
set(gca,'FontSize',15);
[m,delta,pv,hv] = RunningStep(z,Nblock,5,0,'left');
f=find(delta>=jump);
ip=m(f);
hold on;
plot(ip,z(ip),'go','MarkerFaceColor','g');

 subplot(1,2,2)
 plot(F);
 hold on;
 plot(ip,F(ip),'go','MarkerFaceColor','g')
xlabel("index");
ylabel("Force (pN)");
set(gca,'FontSize',15);

pause(0.1);
 analyze=input("Step Good (1=yes) :" );


 if analyze==1
     left=ip(1);
     right=ip(end);
     %[dz(j),sdz(j)] = simple_delta(z,Navg,right,left)
      [dz(j),sdz(j)] = simple_delta_extrapolate(z,Navg,right,left,F,pavg)
     j=j+1;

 end




end

%% Incorporate into total array

dz_all=[dz_all,dz];
sdz_all=[sdz_all,sdz];


%% Compute Distances:


K=600;
lp=50;
lps=2.23;
Lss=340*0.67;
Lds=(2080+90)*0.33;
Fp=1:0.1:40;
ep=WLC_Caliper(Lds,50,Fp,K,Lss,1);
epL=WLC_ext(Lds,50,Fp,600);

Fmeasure=40hist
fm=find(Fp==Fmeasure);
dL_th=ep(fm)-epL(fm);

d_out=dL_th-dz_all;
d_err=sdz_all;