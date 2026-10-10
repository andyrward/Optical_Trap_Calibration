function [fx_TEFVPK1,fp] = fx_OT_Qick_Process_Plot_Cal( ~)
%% Process the raw OT data.
% Output TEFVP (Time, Extension, Force, Voltage, Power) 
% Example: 
% DAT = OT_Qick_Process_Plot('bead3forcejump2.dat','bead3forcejump2_Voltage.dat');
%%031518 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Fixed the time-ordering issue[
% Accounting the Voltage mistype
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Removed the colorline output
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
%%%051818 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Now does the calibration 

% To Do List:

%% Load data
disp("Loading Data");
[path,filedata,fileVoltage] = getfiles()
%[k1, ~] = PSD_Calibration_DY(calfile1);  %051818 edit, still using the AW version
avg_k   = 7e-5;                       %051818 edit good to get more cal files
% avg_k = 5.6371e-05;
x = dlmread(strcat(path,filedata));
x = sortrows(x,1); % issue with save table isn't increasing in time
% x=x(1:927903,:);
d = dlmread(strcat(path,fileVoltage));
PoW = d(1:2:end);
VoT = d(2:2:end); 
State = x (:,6);
fp=strcat(path,filedata);
%% Plot Raw Data
figure;
subplot(3,1,1); plot(x(:,1), x(:,2:5)); ylabel('Tracks (nm)') 
subplot(3,1,2); plot(x(:,1), VoT     ); ylabel('Piezo Voltage (V)')
subplot(3,1,3); plot(x(:,1), PoW     ); ylabel('Laser Power (mW)'); 
                                        xlabel('Time (s)')
disp("raw data");
%% Compute Extension & Force the Filtering out lost track
% range = 1.1760e3:1.3440e3;
% q = (x(range,2)+x(range,4))/2;
disp("Compute Extension and Force")
TiM =  x(:,1);
ExT = (  x(:,3) + x(:,5) - x(:,2) - x(:,4)  )/2;
ExT = ExT - 3000;
FoR = (  x(:,2) + x(:,4)) / 2;
FoR = FoR*avg_k*mean(PoW);
% VoT = VoT';


% force_offset = mean((q).*PoW (range,1) .* avg_k);


%%%

figure;
subplot(3,1,1); plot(TiM, ExT); ylabel('Extension (nm)');
subplot(3,1,2); plot(TiM, FoR); ylabel('Force (~)');
subplot(3,1,3); plot(TiM, VoT); ylabel('Votage (~)');
                                xlabel('Time (s)');  
% 
% 
% figure;
% colormap(jet(256))
% color_line(ExT, FoR, TiM);
% xlabel('Extension (nm)')
% ylabel('Force (~)')

%%%
%%%!!!
Temp = 10;
% % % %%%!!!
% % % 
% LoS = find(ExT> mean(ExT) + Temp*std(ExT) | mean(ExT) - Temp*std(ExT) > ExT);
% TiM(LoS) = []; ExT(LoS) = []; FoR(LoS) = []; VoT(LoS) = []; PoW(LoS) = [];
% LoS = find(VoT > 10);
% TiM(LoS) = []; ExT(LoS) = []; FoR(LoS) = []; VoT(LoS) = []; PoW(LoS) = [];
% Temp = ExT>0;
% ExT = ExT(Temp);
EFV = [ExT FoR VoT];
force_offset = EFV(EFV(:,3)==7.6,:);
force_0=force_offset(:,2);
FoR = FoR- mean(force_0);

%% Checking the time trace
disp("Check time trace")
figure;
subplot(3,1,1); plot(TiM, ExT); ylabel('Extension (nm)');
subplot(3,1,2); plot(TiM, FoR); ylabel('Force (~)');
subplot(3,1,3); plot(TiM, VoT); ylabel('Votage (~)');
                                xlabel('Time (s)'); 
                                
%% Smoothing extension trace
% disp("Smooth extension trace")
% ExT_smooth = smooth(ExT,30);
% FoR_smooth = smooth(FoR,30);
% figure;
% plot(TiM, ExT_smooth); ylabel('Extension (nm)');xlabel('Time (s)');
% figure;
% plot(TiM, FoR_smooth); ylabel('Force (~)');xlabel('Time (s)'); 
% 
% FoR = FoR*956.6*avg_k;
% 

% figure;
% colormap(jet(256))
% color_line(ExT_smooth, FoR_smooth,TiM);
% xlabel('Extension (nm)')
% ylabel('Force (pN)')
% 

% figure;


%% Process Data (TEFVP) Time, Extension, Force, Voltage, Power
disp("process data")
% For some reason.... VoT and Pow can be 1 x m instead of m x 1
% This conver 1 x m to m x 1:
size_VoT = size(VoT); 
if size_VoT(1) < size_VoT(2)
    VoT = VoT';
end

size_PoW = size(PoW); 
if size_PoW(1) < size_PoW(2)
    PoW = PoW';
end
%

fx_TEFVPK1 = [TiM, ExT, FoR, VoT, PoW.*(avg_k+zeros(length(TiM),1)),State]; 


% temp = 1000;
% figure;
% c = clusterdata(TEFVP(1:temp:end,2:4),'maxclust',8);
% scatter3(TEFVP(1:temp:end,2),TEFVP(1:temp:end,3),TEFVP(1:temp:end,4),10,c)
% xlabel('Extension'); ylabel('Force'); zlabel('Piezo Voltage')