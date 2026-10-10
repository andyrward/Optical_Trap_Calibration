
%% Open Data

[data,fp] = fx_OT_Qick_Process_Plot_Cal();


td = array2table(data,'VariableNames',{'t','z','F','V','k','State'});

%% segment data

[ramp, wait] = segment_regions(td.State');

%%
sz=size(ramp);
N=sz(1);
Smax=100; % region must be larger than Smax
zmin=400;
zmax=1300;
sd=size(wait);
fig_l=figure;
c=1;
i=1;
Lss=340*0.64;
Lds=(2080+90)*0.33;
jump = -0.75;
Nblock=40;
Fmeasure=40;
fig_h=figure;
j=1;
d_out=[];
d_err=[];
zu_out=[];
pout=[];
while c==1
figure(fig_l);
    ri=ramp(i,:);

    if(diff(ri)>Smax)

        ru1=ri(1);
        ru2=ri(2);

        if i<=sd(1)
            rwu=wait(i,:);
            rwu1=rwu(1);
            rwu2=rwu(2);
        else
            rwu=0;
        end

        tu=td.t(ru1:ru2);
        Fu=td.F(ru1:ru2);
        zu=td.z(ru1:ru2);
        zu=z_outlier(zu,zmin,zmax);
        if rwu~=0
            tw=td.t(rwu1:rwu2);
            Fw=td.F(rwu1:rwu2);
            zw=td.z(rwu1:rwu2);
            if length(zw)>1
            zw=z_outlier(zw,zmin,zmax);
            end
        end

        in=i+1;

        if in<=N
            ri=ramp(in,:);
            rd1=ri(1);
            rd2=ri(2);

            tdn=td.t(rd1:rd2);
            Fdn=td.F(rd1:rd2);
            zdn=td.z(rd1:rd2);
            if length(zdn)>1
            zdn=z_outlier(zdn,zmin,zmax);
            end


        else
            c=0;
        end



        subplot(3,1,1);
        hold off;
        plot(tu,Fu,'DisplayName',"Ramp Up");
        xlabel('time(sec)');
        ylabel('Force(pN)');
        set(gca,'FontSize',15);
        if rwu~=0
            hold on;
            plot(tw,Fw,'DisplayName',"Wait High Force");

        end
        if c==1
            plot(tdn,Fdn,'DisplayName',"Ramp Down");
        end

        

        subplot(3,1,2);
        hold off;

        plot(zu,Fu,'DisplayName',"Ramp Up");
        xlabel('Extension(nm)');
        ylabel('Force(pN)');
        set(gca,'FontSize',15);

        if c==1
            hold on;
            plot(zdn,Fdn,'DisplayName',"Ramp Down");
            xlim([500,max(zdn)*1.2]);
            ylim([0,max(Fdn)*1.2]);
        end
        i=in+1;

        subplot(3,1,3)
        hold off;
        plot(tu,zu,'DisplayName',"Ramp Up");
        xlabel('time(sec)');
        ylabel('Extension(nm)');set(gca,'FontSize',15);

        pause(0.2);
        analyze=input("Fit Data? (1=yes) :" );

        if analyze==1
            %             [d_out(j),d_err(j),zu_out(j)] = dynamic_delta(zu,zdn,Fu,Fdn,Lds,Lss,Nblock,jump,fig_h,Fmeasure);
            [d_out(j),d_err(j),zu_out(j),pout(j)] = dynamic_delta_fit(zu,zdn,Fu,Fdn,Lds,Lss,Nblock,jump,fig_h,Fmeasure);
            j=j+1;
        end


    else
        i=i+1;

    end

end



%%
z_all=[z_all,zu_out]
d_all=[d_all,d_out]
derr_all=[derr_all,d_err]


