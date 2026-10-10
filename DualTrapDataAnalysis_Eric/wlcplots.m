L1=7308*0.3

z1=0.5*L1:0.001*L1:0.95*L1;
z2=0.5*L2:0.001*L2:0.95*L2;
lp=10
F1=Fwlc(lp,L1,z1);
F2=Fwlc(lp,L2,z2);

figure(3);
subplot(1,2,1);
hold off
plot(F1,z1-z2,'r',F2,z2,'g',F1,z1,'b');
subplot(1,2,2);
plot(F1,(z1-z2)/(L1-L2))