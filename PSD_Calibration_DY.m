function [k1, k2] = PSD_Calibration_DY(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This program allows users to select a file from a browser window with
%either two or three columns of data.  The first column must be time, the
%second must be a position (i.e. left edge), and the third must be a position
%(i.e. right edge) if it exists.

%The second and third rows are averaged to determine the "center position"
%(center position = second row if third row does not exist),and the mean
%is subtracted to obtain the diplacement.  This displacement data is then
%cut into a number of pieces as defined by variable "blocks" and the power
%spectrum of each is taken and averaged.

%An output file is created with "_psd" added to the original filename.cl
%This file now contains frequency in the first column and the averaged
%power spectrum in the second.

% Andy - modified code to calibrate both beads using either 1 or two tracks
% gives back answer in terms of stiffness per power
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Select file from window, load file, and define the dimensions
 [filename,pathname] = uigetfile('*.*','Select file');
 cd(pathname);
file = load(filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dim1 = size(file,1);
dim2 = size(file,2);
f=find(filename=='.');
Vfilename = [filename(1:f-1),'_Voltage.dat'];
Voltage = load(Vfilename);
TrapPower = Voltage(1,1);

display(['Trap Power: ',num2str(TrapPower),'mW']);
%Define variables from input data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%file = sort(file,1); %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file(:,1) = 0;
for I = 1:length(file(:,1))-1 
    if mod(I,2) == 0
        temp =    0.710000000000000*10^-3;
    else
        temp =    0.720000000000000*10^-3;
    end
    file(I+1,1) = file(I,1) + temp;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time = file(:,1);

% TrapNumber=input('Enter Number of Traps to Calibrate: ');
% TrackNumber=input('Enter Number of Tracks for each trap: ');
% AF=input('Manually select fitting region? :');
%

TrapNumber=2
TrackNumber=2
AF=0;

for j=1:TrapNumber
    figure
    switch(TrackNumber)
        case(2)
            left_edge = file(:,j+1);   %should be in nm in original file
            right_edge = file(:,j+3);   %should be in nm in original file
            center = (left_edge + right_edge)/2;
            
            disp_center = (center - mean(center));  %displacement in nanometers
        case(1)
            center = file(:,j+1);
            disp_center = center-mean(center);
    end
    
    
    %Define the number of blocks to be made, take the power spectrum of each
    blocks=64;
    if length(time) < blocks;
        error('The number of blocks is greater than the number of data');
    end
    ndata=length(disp_center)/blocks;
    dt=time(end)/length(time);
    df=1/(ndata*dt);
    for i = 1:blocks;
        ftransform(:,i) = fft(disp_center((i*ndata)-ndata+1:i*(ndata)));
        mag(:,i) = abs(ftransform(:,i));
        psd(1,i) = (mag(1,i).^2)/(df*ndata^2);
        psd((ndata/(2))+1,i) = (mag(((ndata/2)+1),i).^2)/(df*ndata^2);
        psd(2:(ndata/2),i) = (2*(mag(2:(ndata/2),i)).^2)/(df*ndata^2);
    end;
    
    regular_data = interp1(time,disp_center,0:dt:time(dim1)-dt,'spline')';
    for i = 1:blocks;
        ftransform2(:,i) = fft(regular_data((i*ndata)-ndata+1:i*(ndata)));
        mag2(:,i) = abs(ftransform2(:,i));
        psd2(1,i) = (mag2(1,i).^2)/(df*ndata^2);
        psd2((ndata/(2))+1,i) = (mag2(((ndata/2)+1),i).^2)/(df*ndata^2);
        psd2(2:(ndata/2),i) = (2*(mag2(2:(ndata/2),i)).^2)/(df*ndata^2);
    end;
    
    %Define frequency variable
    for k = 1:ndata/2+1;
        freq(k,1) = df*(k-1);
    end
    
    %var1=var(disp_center(1:ndata))
    %var2=sum(psd)*df
    
    %Take the average and plot
    avg_psd = mean(psd,2);
    
    loglog(freq,avg_psd)
    
    %Output data to _psd appended file
    % newdata = [freq avg_psd];
    % savename = strrep(filename,'.dat','_psd2.dat');
    
    % if exist(savename)==2;
    %     overwrite = input('File already exists, overwrite? (y/n)','s');
    %     if overwrite == 'y';
    %         save(savename, 'newdata', '-ascii');
    %     else
    %         uiputfile;
    %     end;
    % else;
    %     save(savename, 'newdata', '-ascii');
    % end;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %This program does a nonlinear fit of the power spectrum
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Initial guesses for gamma and fc, with optional lower and upper bounds
    x0 = [2e-5, 100];
    lb = [1e-6, 0];
    ub = [1e-4, 1000];
    
    %Cut out data that is too noisy
    
    switch(AF)
        
        case(0)
            
            high_pass = (freq < 20);
            filter = (freq > 90) & (120 > freq) | high_pass;
            %filter = (freq > 350) & (400 > freq) | filter;
            %filter = high_pass;
            filter(end) = 1;
            freq_filt = freq(~filter);
            avg_psd_filt = avg_psd(~filter);
            
        case(1)
            
            fb=input('enter frequency low high bounds [fLow, fHigh] ');
            
            fs=find(freq>fb(1) & freq <fb(2));
            freq_filt=freq(fs);
            avg_psd_filt=avg_psd(fs);
            
            
    end
    
    %Run fitting program.  The first argument is the fitting function
    [x_fit,resnorm,residual,exitflag,output] = lsqcurvefit(@filtpsdaliased,x0,freq_filt',avg_psd_filt',lb,ub);
    
    %Display resultsfigure
    x_fit
    
    
    k_fit(j) = 2*pi*x_fit(1)*x_fit(2);
    
    %Plot the data with the fit
    y_fit = filtpsdaliased(x_fit,freq);
    
    loglog(freq,avg_psd,freq,y_fit);
end
display('pN/nm')
k_fit
display('pN/nm/mW');
k1=k_fit(1)/TrapPower

k2=k_fit(2)/TrapPower


