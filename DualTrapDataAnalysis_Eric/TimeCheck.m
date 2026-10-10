[filename,pathname] = uigetfile('*.*','Select file');
cd(pathname);
data = load(filename);
t=data(:,1);
dt=diff(t,1);

figure
subplot(1,2,1);
plot(t);
subplot(1,2,2);
plot(dt);