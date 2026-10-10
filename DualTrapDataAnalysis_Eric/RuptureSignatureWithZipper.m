Lfull=7308*0.3
L1=Lfull-1616*0.3
L2=Lfull-352*0.3

Frupture=40;
Fzip =15;

[F1,z1,d1]=WLC_Mod_DT( lp,L1,K,Frupture,10,Rsum,k1,k2,dstart,0.1 );
[F2,z2,d2]=WLC_Mod_DT( lp,L2,K,Fzip,10,Rsum,k1,k2,d1(end),10 );
[F3,z3,d3]=WLC_Mod_DT( lp,Lfull,K,40,10,Rsum,k1,k2,d2(end),10 );

figure
plot([z1,z2,z3],[F1;F2;F3]);
hold on;
plot(z2,F2,'r');
plot(z3,F3,'g');

