function RESULT2 = OT_Delta_Analysis_Force_pstest(input)
%% RESULT is a table of:
% Timepoint, mean of pre-tran, mean of post-tran, ...
%            numb of pre-tran, numb of post-tran, ...
%            std  of pre-tran, std  of post-tran, ...
%            Voltage when the transition happen, ...
%            mean-force of pre-tran,  mean-f of pre-tran
%
%%%031518 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Removd the "1 sec" Threadhold and changed to 21 ms
% Use the new and improve Plot_ReScale Function: Linear or Sigma
% Fixed the offset of detection location
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% Fixed the issue with the short last voltage
%%%031618 Version: Modification%%%%%%%%%%%%%%%%%%%%%%%%%
% To Do List: Still need to calibrate

%% Data Segmentation base on Voltage
Volt_Inx = find(logical(diff(input(:, 4))) == 1);
Volt_Inx = [0; Volt_Inx; length(input(:, 4))];

%% Alg. for Selecting the zero force offset
temp = sortrows([ unique(input(:,4)), hist(input(:,4), unique(input(:,4)))'], 2,'descend');
temp = sortrows([ unique(input(:,4)), hist(input(:,4), unique(input(:,4)))'], -2);

zero_F_volt = max(temp(1:2, 1));

offset = mean(input( input(:,4) == zero_F_volt,3));

figure;
plot(input(:,3)); hold on;
plot(zeros(length(input(:,3)),1)+offset,'r'); 
plot(zeros(length(input(:,3)),1)+offset+std(input( input(:,4) == zero_F_volt,3)),'r-');
plot(zeros(length(input(:,3)),1)+offset-std(input( input(:,4) == zero_F_volt,3)),'r-'); hold off;


hold off;


%%
test = [];
for I = 1:length(Volt_Inx)-1
    % clc
    % I/(length(Volt_Inx)-1)*100
    st = Volt_Inx(I  ) + 30 + 1;
    ed = Volt_Inx(I+1) - 30;
    
    if st > length(input(:,1))
        st = Volt_Inx(I  ) + 1;
        ed =  length(input(:,1));
    end
    
    if input(st,4) < 7.35
        ch = findchangepts(input(st:ed,2),'MaxNumChanges',1);
        if ~isempty(ch) &&  ch > 1 %
            mi = ch + st;
            test = [ test; mi ...
                mean(input(st:mi,2)) mean(input(mi:ed,2)) ...
                length(st:mi)      length(mi:ed)...
                std(input(st:mi,2))  std(input(mi:ed,2)) ...
                input(mi,4),  ...
                (mean(input(st:mi,3))-offset),...
                (mean(input(mi:ed,3))-offset) ];
            
        end
    end
end
% *input(st,5)
% *input(st,5)
%% Mid-Checking
% clc; close all;
% figure;
% plot(input(:,2)); hold on;
% plot(test(:,1), input(test(:,1),2),'o');
% plot(input(:,4)/(mean(input(:,4)))*3730);
% plot(test(:,1),(test(:,3)-test(:,2))*15+3500,'r.')

%% Ploting of the re-scaled data for Checking
test(isnan(test(:,1)), :) = [];
test(isnan(test(:,2)), :) = [];
figure;
plot(input(:,2)); hold on;
plot(test(:,1), input(test(:,1),2),'o','MarkerSize', 7);
plot(Plot_ReScale(input(:,2),-1*input(:,4), 'empt'));
xlabel('Time Points')

%% Filtering out false transition
test_2 = test;
temp = (test(:,3)-test(:,2));
test_2(temp < 5, :) = []; % Need to add the aditional force criteria



plot(test_2(:,1),input(test_2(:,1),2), 'r.', 'MarkerSize', 14);

%%
figure;
binsize = 25;
dx = test_2(:,3)-test_2(:,2);
temp3 = 3;
LoS = find(dx> mean(dx) + temp3*std(dx) | mean(dx) - temp3*std(dx) > dx);
dx(LoS) = [];
hist(dx,binsize);
xlabel('Delta (nm)'); ylabel('count'); xlim([0 50]);
dx = test_2(:,3)-test_2(:,2);
RESULT2 = [test_2, dx];
% dx=(RESULT2(:,3)-RESULT2(:,2));
% t=(RESULT2(:,1));
% F=(RESULT2(:,9));
% V=(RESULT2(:,8));
% dF=(RESULT2(:,9)-RESULT2(:,10));
% dL=(dF/0.71)+ dx;
% d=38.69-dL;
% 
% 
% For = input(:,3);
% Exten = input (:,2);
% Tim = input(:,1);
% 
% %
% RESULT2 = table(t, V, F, dx, dL, d);

%%
% figure;
% binsize = 25;
% hist(dL, binsize);
% xlabel('DeltaL (nm)'); ylabel('count'); xlim([15 30]);




%% JUNK CODE
% [histw, vinterval] = histwc(test_2(:,3)-test_2(:,2), test_2(:,4).^2+(test_2(:,5).^2),binsize);
% subplot(2,1,2);bar(vinterval, histw)
% xlabel('Delta (nm)'); ylabel('frequency');  xlim([27 34])
%
%      plot(test(:,3)-test(:,2), min(test(:,5),test(:,4)),'.')
%
%      gm   = fitgmdist(data,2);
%      idx  = cluster(gm,data);
%      idx  = find( logical(diff(idx) == 1 ))
%      test = [test; idx+Volt_Inx(I)+1];
%
%      MinPts = 700;
%      inc    = 1;
%      while 1
%          [IDX, isnoise]=DBSCAN(data,0.05, inc * MinPts);
%          if length(IDX) > 2
%              inc = inc + 1;
%          else
%              break
%          end
%      end
%      test = [test; IDX];
%      inx = kmeans(smooth(data, smW),2);
%      test = [test;find( logical(diff(inx)==1))+Volt_Inx(I)+1];