%extraction and separation of the data
tim = fx_TEFVPK1(:,1);
ext = fx_TEFVPK1(:,2);
f = fx_TEFVPK1(:,3);
vol = fx_TEFVPK1(:,4);

%segmentation of the data based on voltage cycle
int = findchangepts(vol,'Statistic','linear','MinThreshold',500);

% determine dx of shearing

esh = [];

for i = 1:length(int)
        ei = ext(int(i)-3500:int(i)-20);
    ti = tim(int(i)-3500:int(i)-20);
    fi = f(int(i)-3500:int(i)-20);
    vi = vol(int(i)-3500:int(i)-20);
    tefvi =  [ti ei fi vi];

figure(1); hold on
subplot(5,6,i);hold on
plot(ti,ei); hold on
    
temp1 = vi == 6.45;
temp2 = vi == 6.4;
TEFVsh1 = tefvi(temp1,:);
TEFVsh2 = tefvi(temp2,:);
timsh1 = TEFVsh1(10:end-10,1);
extsh1 = TEFVsh1(10:end-10,2);
fsh1 = TEFVsh1(10:end-10,3);
timsh2 = TEFVsh2(10:end-10,1);
extsh2 = TEFVsh2(10:end-10,2);
fsh2 = TEFVsh2(10:end-10,3);
plot(timsh1,extsh1,'.r');
plot(timsh2,extsh2,'.r');
plot(mean(timsh1), mean(extsh1), '.b',"MarkerSize",10);
plot(mean(timsh2), mean(extsh2), '.b',"MarkerSize",10);

esh = [esh; mean(fsh1) mean(fsh2) mean(extsh2)-mean(extsh1) 41.38-(mean(extsh2)-mean(extsh1))];
end

%%

temp3 = esh(:,3)>50 & esh(:,3)<95;
esh1 = esh(temp3,:);
figure;
histogram(esh1(:,3), 20);