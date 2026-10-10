function [TEFVPK1] = OT_Qick_Process_Plot_Cal(filedata, fileVoltage, calfile)
%% Process the raw OT data.
% Output TEFVP (Time, Extension, Force, Voltage, Power) 
% Example: 
% DAT = OT_Qick_Process_Plot('bead3forcejump2.dat','bead3forcejump2_Voltage.dat');
%%031518 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Fixed the time-ordering issue
% Accounting the Voltage mistype
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Removed the colorline output
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
%%%051818 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Now does the calibration 

% To Do List:

%% Load data

[k1, ~] = PSD_Calibration_DY(calfile);  %051818 edit, still using the AW version
avg_k   = k1;                       %051818 edit good to get more cal files

x = dlmread(filedata);
x = sortrows(x,1); % issue with save table isn't increasing in time
d = dlmread(fileVoltage);
PoW = d(1:2:end);
VoT = d(2:2:end);

%% Plot Raw Data
figure;
subplot(3,1,1); plot(x(:,1), x(:,2:5)); ylabel('Tracks (nm)') 
subplot(3,1,2); plot(x(:,1), VoT     ); ylabel('Piezo Voltage (V)')
subplot(3,1,3); plot(x(:,1), PoW     ); ylabel('Laser Power (mW)'); 
                                        xlabel('Time (s)')

%% Compute Extension & Force the Filtering out lost track
% range = 1.1760e4:1.3440e4;
% q = (x(range,2)+x(range,4))/2;
TiM =  x(:,1);
ExT = (  x(:,3) + x(:,5) - x(:,2) - x(:,4)  )/2;
FoR = (  x(:,2) + x(:,4)) / 2;
% force_offset = mean((q).*PoW (range,1) .* avg_k);
% FoR = force- force_offset;

%%%!!!
Temp = 1.2;
%%%!!!

LoS = find(ExT> mean(ExT) + Temp*std(ExT) | mean(ExT) - Temp*std(ExT) > ExT);
TiM(LoS) = []; ExT(LoS) = []; FoR(LoS) = []; VoT(LoS) = []; PoW(LoS) = [];
LoS = find(VoT > 10);
TiM(LoS) = []; ExT(LoS) = []; FoR(LoS) = []; VoT(LoS) = []; PoW(LoS) = [];

%% Checking the time trace
figure;
subplot(3,1,1); plot(TiM, ExT); ylabel('Extension (nm)');
subplot(3,1,2); plot(TiM, FoR); ylabel('Force (~)');
subplot(3,1,3); plot(TiM, VoT); ylabel('Votage (~)');
                                xlabel('Time (s)');  

%% Checking the Force-Extension Curve
figure;
colormap(jet(256))
color_line(smooth(ExT,10), smooth(FoR,10),TiM);
xlabel('Extension (nm)')
ylabel('Force (~)')

%% Process Data (TEFVP) Time, Extension, Force, Voltage, Power
TEFVPK1 = [TiM, ExT, FoR, VoT, PoW.*(avg_k+zeros(length(TiM),1))]; 


% temp = 1000;
% figure;
% c = clusterdata(TEFVP(1:temp:end,2:4),'maxclust',8);
% scatter3(TEFVP(1:temp:end,2),TEFVP(1:temp:end,3),TEFVP(1:temp:end,4),10,c)
% xlabel('Extension'); ylabel('Force'); zlabel('Piezo Voltage')