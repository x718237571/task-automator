clc;clear ; close all
% tic









%% Generate normally distributed random numbers by LHS
n = 100; % Number of samples
p = 5; % Number of variables (dimensions)

% Generate LHS matrix with uniform distribution
lhs_uniform = lhsdesign(n, p);

% Initialize matrix for normally distributed samples
lhs_normal = zeros(size(lhs_uniform));

% Define the mean and standard deviation for each dimension
mu = [10, 20, 30, 40, 50]; % Example means for each variable
sigma = [2, 4, 6, 8, 10]; % Example standard deviations for each variable

% Transform the uniform samples to normal distribution
for i = 1:p
    lhs_normal(:, i) = norminv(lhs_uniform(:, i), mu(i), sigma(i));
end

% lhs_normal now contains the LHS samples from a normal distribution

hist(lhs_normal(:, 1))

%% 读log.txt文件里UnStable的个数
clc; clear; close all
% str1 = fileread('C:\Users\Yin\Desktop\log1.txt');
% str2 = fileread('C:\Users\Yin\Desktop\log2.txt');
% str = [str1 str2];
str = fileread('C:\Users\Yin\Desktop\log.txt');
n_all = length(strfind(str,'Stable'));
n_fail = length(strfind(str,'UnStable'));
fprintf('%d out of %d failed.\n', n_fail, n_all);

Pf = n_fail/n_all;
COV_Pf = sqrt((1-Pf)./(n_all*Pf));
fprintf('Pf is %.4f, with a COV of %.3f.\n', Pf, COV_Pf);

%% Plot 2D random field
clc;clear ; close all
pos=importdata('C:\Users\Yin\Desktop\10m_slope\FLAC\CentroidPosition.txt');
pos=ceil(pos*2);
rfdata=importdata('C:\Users\Yin\Desktop\Karhunen Loeve\Same_RF_diff_trunc\data\auto_exp2\cohesion68.txt');
c=rfdata(:,1);

% scatter(pos(:,1),pos(:,2))

for i=1:length(pos)
    C(pos(i,2),pos(i,1))=c(i);
end

% c = colorbar;c.Label.String = 'Grayscale';
% colormap('gray');
axis off
C(find(C==0))=nan;        % 只关心边坡区域，把剩下为0的地方都删为NAN

% [F1,ps]=mapminmax(F,0,1);
h=pcolor(C);set(gca,'YDir','normal');set(gcf,'position',[1000,950,750,250]);
% colorbar('fontsize',12);
set(h, 'EdgeColor', 'none'); % [.6 .6 .6] / 'none' 别漏单引号
% xlabel('Length(m)');ylabel('Depth(m)');
xticks([]);yticks([]);
% caxis([11 34])
% imagesc(C)
% set(gca,'YDir','normal')
mean(rfdata,'all')
std(reshape(rfdata,numel(rfdata),1))

axis tight;  % This ensures that the axes tightly fit the plot
% set(gca,'LooseInset',get(gca,'TightInset'));  % This removes the space around the plot
set(gca, 'Color', 'none');  % Makes figure background transparent
set(gca, 'YColor', 'none');  % remove edge
set(gca, 'XColor', 'none');
box off
%%%%%%%%%%%%%% Save the figure as an EMF without a border %%%%%%%%%%%%%%%%
% print -dmeta -r0 'RandomField.emf'
% print -dpng -r300 'RandomField.png'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1D坐标转2D（KL-RF的画图用）4层坡 Congress Street Cut
clc;clear;close all

pos0=importdata('C:\Users\Yin\Desktop\Congress street cut\FLAC\cen_xz_sand.txt');
pos0=ceil(pos0*2);
pos1=importdata('C:\Users\Yin\Desktop\Congress street cut\FLAC\cen_xz_clay1.txt');
pos1=ceil(pos1*2);
pos2=importdata('C:\Users\Yin\Desktop\Congress street cut\FLAC\cen_xz_clay2.txt');
pos2=ceil(pos2*2);
pos3=importdata('C:\Users\Yin\Desktop\Congress street cut\FLAC\cen_xz_clay3.txt');
pos3=ceil(pos3*2);

rfdata=importdata('C:\Users\Yin\Desktop\Congress street cut\DataPairs\low\RFdata\cphi2.txt');
c=rfdata(:,1);
f=rfdata(:,2);

L1C=c(1:length(pos1));
L2C=c(length(pos1)+1:length(pos1)+length(pos2));
L3C=c(length(pos1)+length(pos2)+1:length(pos1)+length(pos2)++length(pos3));
L1F=f(1:length(pos1));
L2F=f(length(pos1)+1:length(pos1)+length(pos2));
L3F=f(length(pos1)+length(pos2)+1:length(pos1)+length(pos2)++length(pos3));

% scatter(pos0(:,1),pos0(:,2))
% hold on
% scatter(pos1(:,1),pos1(:,2))
% hold on
% scatter(pos2(:,1),pos2(:,2))
% hold on
% scatter(pos3(:,1),pos3(:,2))

for i=1:length(pos1)
    C(pos1(i,2),pos1(i,1))=L1C(i);
end
for i=1:length(pos2)
    C(pos2(i,2),pos2(i,1))=L2C(i);
end
for i=1:length(pos3)
    C(pos3(i,2),pos3(i,1))=L3C(i);
end
C(find(C==0))=nan;        % 只关心边坡区域，把剩下为0的地方都删为NAN

h=pcolor(C);set(gca,'YDir','normal');set(gcf,'position',[800,900,900,270]);
colorbar('fontsize',12);
set(h, 'EdgeColor', 'none'); % [.6 .6 .6] / 'none' 别漏单引号
xlabel('Length(m)');ylabel('Depth(m)');

xticks([]);yticks([]);
axis off

% c = colorbar;c.Label.String = 'Grayscale';
% colormap('gray');


% for i=1:length(pos1)
%     F(pos1(i,2),pos1(i,1))=L1F(i);
% end
% for i=1:length(pos2)
%     F(pos2(i,2),pos2(i,1))=L2F(i);
% end
% figure
% imagesc(F)
% set(gca,'YDir','normal')

% 查看坐标排列顺序
% for i=1:length(pos1)
%     scatter(pos1(i,1),pos1(i,2));
%     pause(1/1000)
%     hold on
% end

%% Prepare the active learning test data
Num_train = 847*2+1000*3;
pos0=importdata('D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC\CentroidPosition.txt'); % ..\..\ upper upper folder
pos=ceil(pos0*2);
train_c =  load('C:\Users\Yin\Desktop\cohesion.txt');                % cohesion random fields
train_phi = load('C:\Users\Yin\Desktop\friction.txt');             % friction random fields
load('C:\Users\Yin\Desktop\pilepos.mat');                       % pile position
% Input (Samples)
for i = 1:847
    rf_c=train_c(:,1);
    rf_phi=train_phi(:,1);
    input1(:,:,1,i) = input_image(rf_c,pos);
    input1(:,:,2,i) = input_image(rf_phi,pos);
    input1(:,:,3,i) = pposition(:,:,i);  % attention
end
for i = 1:847
    rf_c=train_c(:,2);
    rf_phi=train_phi(:,2);
    input2(:,:,1,i) = input_image(rf_c,pos);
    input2(:,:,2,i) = input_image(rf_phi,pos);
    input2(:,:,3,i) = pposition(:,:,i);  % attention
end
for i = 1:1000
    rf_c=train_c(:,i);
    rf_phi=train_phi(:,i);
    input3(:,:,1,i) = input_image(rf_c,pos);
    input3(:,:,2,i) = input_image(rf_phi,pos);
    input3(:,:,3,i) = pposition(:,:,331);  % attention
end
for i = 1:1000
    rf_c=train_c(:,i);
    rf_phi=train_phi(:,i);
    input4(:,:,1,i) = input_image(rf_c,pos);
    input4(:,:,2,i) = input_image(rf_phi,pos);
    input4(:,:,3,i) = pposition(:,:,146);  % attention
end
for i = 1:1000
    rf_c=train_c(:,i);
    rf_phi=train_phi(:,i);
    input5(:,:,1,i) = input_image(rf_c,pos);
    input5(:,:,2,i) = input_image(rf_phi,pos);
    input5(:,:,3,i) = pposition(:,:,588);  % attention
end
input_database=cat(4,input1,input2,input3,input4,input5);
input_database(:,:,1,:)=normdata(input_database(:,:,1,:), 2, 41, 0, 1); % min & max
input_database(:,:,2,:)=normdata(input_database(:,:,2,:), 8, 47, 0, 1);
input_database(input_database<0)=0;input_database(input_database>1)=1;

%% Four types of Label database
clc;clear;close all

inhomo_nopile = load('D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC\inhomo-nopile\CalculatedFoS.csv');

for i=1:5
    inh_ps(:,i) = load(sprintf('D:/202210_surrogate_pile_RBDO/FLAC_struct2/FLAC/inhomo-piles/rf%d/CalculatedFoS.csv',i));
end
inhomo_piles = reshape(inh_ps,[numel(inh_ps),1]);

homo_piles = load('D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC\homo-piles\FLAC1&2\CalculatedFoS_2nd.csv');
% all database fs
train_fs = [inhomo_nopile(1:100,1); inhomo_1pile; inhomo_piles; homo_piles];
writematrix(train_fs,'train_fs.csv');
%% Prepare inhomo_1pile fs in 17 position
sdir='D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC';
cd(sdir);
file=dir();
for i=3:19
    p_pos=file(i).name; % folder's name (its position)
    datadir=fullfile(sdir,p_pos,'CalculatedFoS.csv'); % directory
    pdata=load(datadir); % fs data
    p_pos=strrep(p_pos,'.','_'); % replace '.'
    p_pos=strrep(p_pos,'-','m'); % replace '.'
    p_name{i-2,1}=['fs',p_pos]; % variable name
    eval([p_name{i-2,1},'=pdata;']); % Assign values
    Pf(i-2)=length(pdata(pdata<1))/length(pdata);
end
inhomo_1pile=[fs10m0(1:100,1)
fs10m5(1:100,1)
fs13_75m0(1:100,1)
fs13_75m10(1:100,1)
fs13_75m5(1:100,1)
fs17_5m0(1:100,1)
fs17_5m10(1:100,1)
fs17_5m5(1:100,1)
fs21_25m0(1:100,1)
fs21_25m10(1:100,1)
fs21_25m15(1:100,1)
fs21_25m5(1:100,1)
fs25m0(1:100,1)
fs25m10(1:100,1)
fs25m15(1:100,1)
fs25m17_5(1:100,1)
fs25m5(1:100,1)
 ];

%% Predict Pf
tic
Num_test=5000;
load('.\pposition18.mat');
pf_c =  load('.\pf_c.txt');                % cohesion random fields
pf_phi = load('.\pf_phi.txt');             % friction random fields
for j = 1: Num_test
    rf_c=pf_c(:,j);   % attention 100+j
    rf_phi=pf_phi(:,j);   % attention
    predict_input(:,:,1,j) = input_image(rf_c,pos);
    predict_input(:,:,2,j) = input_image(rf_phi,pos);
    predict_input(:,:,3,j) = pposition18(:,:,14);  % attention
end
predict_input(:,:,1,:)=normdata(predict_input(:,:,1,:), 2, 41, 0, 1); % min & max 
predict_input(:,:,2,:)=normdata(predict_input(:,:,2,:), 8, 47, 0, 1);
predict_input(predict_input<0)=0; predict_input(predict_input>1)=1; % mainly for trapezoid
% Predict
FS_Net1 = predict(Net1,predict_input); 
FS_Net2 = predict(Net2,predict_input);
FS_Net3 = predict(Net3,predict_input);
FS_Net4 = predict(Net4,predict_input);
FS_Net5 = predict(Net5,predict_input);

Pf_Net1 = sum(FS_Net1(:,1)<1)/Num_test;
Pf_Net2 = sum(FS_Net2(:,1)<1)/Num_test;
Pf_Net3 = sum(FS_Net3(:,1)<1)/Num_test;
Pf_Net4 = sum(FS_Net4(:,1)<1)/Num_test;
Pf_Net5 = sum(FS_Net5(:,1)<1)/Num_test;
Pf_ave = mean([Pf_Net1 Pf_Net2 Pf_Net3 Pf_Net4 Pf_Net5])
toc
%% prepare input database of pile position (Cai 2000)
clc;clear;close all
% px= [10 10 13.75 13.75 13.75 17.5 17.5 17.5 21.25 21.25 21.25 21.25 25 25 25 25   25];
% py1=[0  5  0     10    5     0    10   5    0     10    15    5     0  10 15 17.5 5 ];
pile_pos=readmatrix('C:\Users\Yin\Desktop\pile.xlsx');
px=pile_pos(:,3);py1=pile_pos(:,4);

pnum=length(px);
pposition=zeros(40,70,pnum);
for i=1:pnum
    pposition_npile(:,:,i)=ppdata40x70(px(i),py1(i),20);
%     pposition(:,:,i)=ppdata40x70(px(i),py1(i),20);
%     writematrix(pposition40x70(:,:,i),'pposition40x70.xlsx','Sheet',i);
end
pposition18(:,:,1)=zeros(40,70);
pposition18(:,:,2:pnum+1)=pposition_npile;

% save('pposition18.mat','pposition18')
save('pilepos.mat','pposition')
% [px' py1']; % check order
%% test ANN
clc;clear ; close all
% [x1,t1] = simplefit_dataset;

syms x
f(x)=1.5*x*sin(x)+1.3*x;
x1=0:0.1:14;
t1=eval(f(x1));
plot(x1,t1);hold on;

randind=randperm(length(x1));
n=20;
x=x1(randind(1:n));t=t1(randind(1:n));
scatter(x,t,'r*');hold on;

% net = fitnet(10);
% net = train(net,x,t); % train
% y1 = net(x1); % predict

net = fitnet(10,'trainbr');

net = train(net,x,t); % train
% loss = myLoss(Y,T);
y1 = net(x1); % predict

[x2,ind]=sort(x);
plot(x1,y1,'.','color','#D95319');
legend('Line','ANN','Samples','location','northwest')
%% Lognormal probability distribution
clc;clear ; close all

m = 20; % mean
v = 6; % variance
mu = log((m^2)/sqrt(v+m^2));
sigma = sqrt(log(v/(m^2)+1));
% rng('default') % For reproducibility

a=lognrnd(mu,sigma,[10000,1]);
histogram(a)
%% kstest2
clc;clear ; close all
rng(1);     % For reproducibility
x1 = wblrnd(1,1,1,50);
x2 = wblrnd(1.2,2,1,50);

h = kstest2(x1,x2)
cdfplot(x1)
%% 1D to 2D (FS contour)
clc;clear;close all

% read the fs sensitivity map data
c = readmatrix('D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC\inhomo-piles\rf2\CalculatedFoS.csv');
pile = readmatrix('D:\202210_surrogate_pile_RBDO\FLAC_struct2\FLAC\homo-piles\pile.xlsx');
pos = pile(:,3:4); 

for i=1:length(pos)
    C(pos(i,2)+1,pos(i,1))=c(i,1); 
end
C(C==0)=nan;

% for i=1:200
h=pcolor(C); % (:,10:25)
set(gca,'YDir','reverse')
caxis([0.93 1.22])
colorbar

set(gca,'YDir','normal');set(gcf,'position',[1000,750,650,350]);
set(gca,'YDir','normal')
set(h, 'EdgeColor', 'none'); % [.6 .6 .6] / 'none' 别漏单引号
axis off

% saveas(gcf, sprintf('C:/Users/Yin/Desktop/colormap/pic%d.png',i))
% end
%% 用kriging插值函数测试 优化
clc;clear;close all
tic
x0 = [18,5.5];
A = [-1,2];
b = 1;
Aeq = [];
beq = [];
lb = [10,1];
ub = [20,11];
nonlcon = [];
options = optimset('PlotFcns',@optimplotx); % 'Display','iter'
x=fmincon(@pf_cal,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)

% x = ga(@pf_cal,2,A,b,Aeq,beq,lb,ub,nonlcon)
% 在optimplotx里加个x输出
toc
%% Kriging CNN Pf
clc;clear ; close all
load('D:\202210_surrogate_pile_RBDO\Meta\cnn1L\allPf.mat');
z=Pf;
p0=readmatrix('D:\202210_surrogate_pile_RBDO\FLAC_struct\5 firm stratum2\pile.xlsx');
p1=[p0(:,3) p0(:,4)+1];
n=100;
%模型参数设置，无特殊情况不需修改，见说明书
theta = [0.1 0.1]; lob = [1e-1 1e-1]; upb = [20 20];
%变异函数模型为高斯模型
[dmodel, perf] = dacefit(p1, Pf, @regpoly0, @correxp, theta); %   ,lob, upb 
%创建一个40*40的格网，标注范围为0-100，即格网间距为2.5
%S存储了点位坐标值，Y为观测值
X = gridsamp([9 1;21 12], n);     
% X=[83.731	32.36];     %单点预测的实现
%格网点的预测值返回在矩阵YX中，预测点的均方根误差返回在矩阵MSE中
[YX,MSE] = predictor(X, dmodel);    
X1 = reshape(X(:,1),n,n); X2 = reshape(X(:,2),n,n);
YX = reshape(YX, size(X1));         %size(X1)=40*40
figure(1), surfc(X1, X2, YX)         %绘制预测表面
shading interp
hold on,
plot3(p1(:,1),p1(:,2),z,'.k', 'MarkerSize',10)    %绘制原始散点数据
hold off
% figure(2),mesh(X1, X2, reshape(MSE,size(X1)));  %绘制每个点的插值误差大小
%% plot CNN Pf contour
clc;clear;close
data=readmatrix('C:\Users\Yin\Desktop\fs_contour\fos.xlsx','Sheet','FS_FLAC3rd');
z = data(:,3);

x=data(:,1);y=data(:,2);
[X,Y]=meshgrid(10:0.5:25,0:0.5:20);
Z=griddata(x,y,z,X,Y,'cubic');
surfc(X,Y,Z); set(gcf,'position',[250,600,600,400]);
xlabel('Distance');ylabel('Elevation');zlabel('Pf');
shading interp
caxis([1.1 1.62])
colorbar
grid off
% save gif
% for i = 1:360
%     camorbit(1,0,'data',[0 0 1])
%     %    drawnow
%     M=getframe(gcf);
%     nn=frame2im(M);
%     [nn,cm]=rgb2ind(nn,256);
%     if i==1 % i==1 loopcount works
%         imwrite(nn,cm,'out.gif','gif','LoopCount',inf,'DelayTime',0.1);
%     else % i>=2 loopcount not works
%         imwrite(nn,cm,'out.gif','gif','WriteMode','append','DelayTime',0.1) 
%     end
% end

%% Obtain some Pfs
tic
p0=readmatrix('C:\Users\Yin\Desktop\pile.xlsx');
p1=[p0(:,3) p0(:,4)+1];
for i=1:length(p1)
    Pf(i,1)=calculatePf(p1(i,:));
end
toc
%% Prepare training database
% random fields
data4=readmatrix('D:\202210_surrogate_pile_RBDO\FLAC_struct\5 firm stratum2\RF_data4_200.xlsx');
dataPf=readmatrix('D:\202210_surrogate_pile_RBDO\FLAC_struct\5 firm stratum2\RF_for_Pf_200.xlsx');
testrf=[repmat(data4(:,101:200),1,10) repmat(dataPf(:,101:200),1,5) repmat(data4(:,101:200),1,2)];
writematrix(testrf,'test_rf.csv');
% position
load('D:\202210_surrogate_pile_RBDO\Meta\cnn1L\pposition22x60.mat');
pposition2=pposition(:,:,:);
save('pposition22x60test.mat','pposition2')
% fs
testfs=[fs0(101:200)
fs20(101:200)
fs10(101:200)
fs12_5(101:200)
fs15(101:200)
fs17_5(101:200)
fs18(101:200)
fs16(101:200)
fs13_75(101:200)
fs12(101:200)
fs17_5m3_5(101:200)
fs15m3_5(101:200)
fs17_5m6(101:200)
fs12_5m6(101:200)
fs15m6(101:200)
fs17_5m8_5(101:200)
fs20m8_5(101:200) ];
writematrix(testfs,'test.csv');

%% save pile position to excel sheets (for training input)
for i=1:21
    data=pposition(:,:,i);
    writematrix(data,'position.xlsx','sheet',i);
end
% check tomorrow
%% plot Pf contour by FDM
x=[10;10;12;12.5;12.5;12.5;13.75;15;15;15;16;17.5;17.5;17.5;17.5;18;20;20;20;20];
y=[1;3.5;1;1;2.5;6;1;1;3.5;6;1;1;3.5;6;8.5;1;1;3.5;6;8.5];
% 20 pile position mean fs
z=[1.29;1.20;1.31;1.33;1.26;1.21;1.35;1.36;1.23;1.21;1.35;1.38;1.23;...
    1.21;1.23;1.37;1.38;1.26;1.19;1.22];
[X,Y]=meshgrid(9:0.5:21,0:0.5:11);
Z=griddata(x,y,z,X,Y);
surf(X,Y,Z); set(gcf,'position',[250,600,600,400]);
colorbar
title('FS');xlabel('Distance/m');ylabel('Elevation/m');
% scatter(X,Y)
%% load fs results at different pile position
clc;clear ; close all
% sdir=uigetdir('Choose source directory.');
sdir='D:\202210_surrogate_pile_RBDO\FLAC_struct\5 firm stratum2';
cd(sdir);
file=dir();
% file_cell = struct2cell(file);
% fs_all=[];
for i=3:length(file) % 2 folders should not be considered
    if isfolder(file(i).name)==1 % find the folder
        p_pos=file(i).name; % folder's name (its position)
        datadir=fullfile(sdir,p_pos,'CalculatedFoS.csv'); % directory
        pdata=load(datadir); % fs data
        p_pos=strrep(p_pos,'.','_'); % replace '.'
        p_pos=strrep(p_pos,'-','m'); % replace '.'
        p_name{i,1}=['fs',p_pos]; % variable name
        eval([p_name{i,1},'=pdata;']); % Assign values
        Pf(i)=length(pdata(pdata<1))/length(pdata);
%         fs_all=[fs_all pdata]; % save fs data in a matrix
    end
end
p_name(cellfun(@isempty,p_name))=[];
% plot(mean(fs_all))
% for i = 1:9
%     ind(i)=length(find(fs_all(:,i)<1));
% end
for i=1:length(p_name)
    mu(i,1)=mean(eval(p_name{i,1}));
    sd(i,1)=std(eval(p_name{i,1}));
end
%% plot fs contour
clc;clear ; close all
C = readmatrix('C:\Users\yin\Desktop\str\pile.xlsx','Sheet','Sheet1');
FS = readmatrix('C:\Users\yin\Desktop\fs_sensitivity\CalculatedFoS.csv');

x=C(:,3);
y=C(:,4);
z=FS;
[X,Y]=meshgrid(9:0.5:21,0:0.5:12);
Z=griddata(x,y,z,X,Y);
surf(X,Y,Z); set(gcf,'position',[250,600,600,400]);
% mesh(X,Y,Z);plot3(x,y,z,'.b')%显示离散点本身
figure
surf(X,Y,Z);shading interp;colormap parula;colorbar;set(gcf,'position',[900,600,600,400]);
hold on;
M=contour(X,Y,Z,5);%显示等高线
tab=tabulate(C(:,1));
count=tab(:,2);count=count(count~=0);
% unique(FS);

k=1;
for j=1:length(count)
    FS1(1:count(j),j)=FS(k:k+count(j)-1,1);
    k=k+count(j);
end
FS2=flipud(FS1);
FS2(FS2==0)=nan;
figure
imagesc(FS2);colorbar;set(gcf,'position',[10,100,600,400]);
FS3=[FS2(1,:);FS2];FS3=[FS3 FS3(:,end)];
figure
pcolor(flipud(FS3));colorbar;set(gcf,'position',[650,100,600,400]);
figure
imagesc(FS2(1:end-1,:));colorbar;set(gcf,'position',[1300,100,600,400]);

%% generate FISH txt with different piles struct
% type template.txt
clc;clear ; close all
dir=uigetdir('Choose source directory.'); % include 0 template & pile
fid=fopen(fullfile(dir,'\0 template.txt'));
line=' ';
i=1;
for i=1:100
    line = fgetl(fid);
    if line==-1
        break
    end
    str{i,1}=line;
    
end
fclose(fid);

M = readmatrix(fullfile(dir,'\pile.xlsx'),'Sheet','Sheet1');
format='struct beam create by-line (%g,0,%g) (%g,0,%g) segments=%d';
for i=1:size(M,1)
    n1=M(i,1);n2=M(i,2);n3=M(i,3);n4=M(i,4);n5=ceil((n2-n4)*1.8); %nodes num 
    str_pile{i,1}=sprintf(format,n1,n2,n3,n4,n5);
end
newdir = strrep(dir,'\','/');
for j=1:size(str_pile,1)
    str{24}=str_pile{j}; % attention for the position
    fid = fopen(sprintf(strcat(newdir,'/geometry%d.f3dat'),j), 'w'); % strcat avoid fullfile '\'
    for i = 1:size(str, 1)
        fprintf(fid, '%s\n', str{i,:});
    end
    fclose(fid);
end

%% generate pile top and bottom position
clc;clear;close all
dir=uigetdir('Choose source directory.'); % for saving pile
step=0.5;
p=1; % 3rd dimension of matrix
for x=10:step:25       % slope range
    coor(1,2,p)=(2*x+10)/3;   % pile top
    coor(1,4,p)= 0;
    row=1; % row
    while coor(row,4,p) < (2*x+10)/3-2
        coor(row+1,4,p)= coor(row,4,p)+step;
        row=row+1;
    end
    coor(:,1,p)=x; coor(:,2,p)=coor(1,2,p); coor(:,3,p)=x;
    p=p+1;
end

for i=1:size(coor,3)
    A(:,i)=coor(:,4,i);
end
A=flipud(A); % the pile bottom in 2D

index=find(coor(:,1,:)==0);
all=[];jj=1;
for j=1:size(coor,3)
    all=[all;coor(:,:,j)];
    jj=jj+1;
end
all(index,:)=[];
% stratum=zeros(size(all));
% stratum(:,2)=1;stratum(:,4)=1;
% all2=all+stratum;

writematrix(all,fullfile(dir,'\pile.xlsx')); % save coor

c=[all(:,1:2);all(:,3:4)];
scatter(c(:,1),c(:,2));
%% This is txt name changing codes I did not succeed
srcDir=uigetdir('Choose source directory.'); %获得选择的文件夹
cd(srcDir); %更改根路径
file = dir('*.txt');
len = length(file);

for i = 1 : len
   oldname0 = string(file(i).name);
   oldname = strcat(',',oldname0);
   Date=string(regexp(oldname,'.*(?=\.txt)','match')); % 正则表达式
   newname = strcat(Date,'.f3dat');
   % the following sentence does not work:
   eval(['!rename',char(oldname),char(newname)]); %要用char  string不行
end

% so what to do:
%{
Win+R:cmd
cd C:\Users\yin\Desktop\str
dir /s
ren *.txt *.f3dat
%}

% actually save .f3dat is OK rather than .txt
%% 预测未训练桩位fos
numofexm=400;                                  % attention
ZFoS_ConvNet1 = zeros(numofexm,1);
ZFoS_ConvNet2 = zeros(numofexm,1);
ZFoS_ConvNet3 = zeros(numofexm,1);
ZFoS_ConvNet4 = zeros(numofexm,1);
ZFoS_ConvNet5 = zeros(numofexm,1);

for i = 1:numofexm
    rfdata=importdata(sprintf('C:/Users/Yin/Desktop/Meta/RF_data3/rf%d.txt',i));% attention
    cohesion_store = input_image(rfdata,pos);
    img2predict(:,:,1) = cohesion_store(:,:);
    img2predict(:,:,2) = pposition1(:,:,ceil(i/100));   % attention
    db1=normdata(img2predict(:,:,1), 5, 80, 0, 1); % attention min & max should be determine
    db1(find(db1<0))=0;db1(find(db1>1))=1;
    img2predict(:,:,1)=db1;

    ZFoS_ConvNet1(i,1) = predict(ConvNet1,img2predict);
    ZFoS_ConvNet2(i,1) = predict(ConvNet2,img2predict);
    ZFoS_ConvNet3(i,1) = predict(ConvNet3,img2predict);
    ZFoS_ConvNet4(i,1) = predict(ConvNet4,img2predict);
    ZFoS_ConvNet5(i,1) = predict(ConvNet5,img2predict);
end

Z_tureFoS = load('C:\Users\Yin\Desktop\FLAC_struct\2 flac3d\train_data\tfos.csv');
%% 调用Pf计算的子程序进行设计参数更新
tic
x0 = [17,8];
A = [-1,2];
b = 1;
Aeq = [];
beq = [];
lb = [10,1];
ub = [20,11];
nonlcon = [];
options = optimset('PlotFcns',@optimplotx); % 'Display','iter'
x=fmincon(@calculatePf,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
% 在optimplotx里加个x输出
toc
%% 测试fmincon优化函数
clc;clear;close all
func=@(x)(x(1))^2+(x(2))^2;

x0 = [-18,15];
A = [1,1];
b = 50;
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = [];
options = optimset('PlotFcns',@optimplotx); % 'Display','iter'
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(func,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
clear A b Aeq beq lb ub nonlcon options

% 在optimplotx里改出个x输出就行
%% 画个优化问题用的函数图
clc;clear;close all
x=-5:.2:5; y=-5:.2:5;
[X,Y]=meshgrid(x,y);
X1=reshape(X,[numel(X),1]);Y1=reshape(Y,[numel(Y),1]);
xx=[X1 Y1];
func=@(x)(x(1))^2+(x(2))^2
for i=1:numel(X)
Z(i)=func(xx(i,:));
end
Z=reshape(Z,[length(x),length(y)]);
surf(X,Y,Z)
%% prepare input database of pile position
clc;clear;close all
px=[20 10 12.5 15 17.5 18 16 13.75 12 12.5 20 20 10 17.5 15 17.5 12.5 15 17.5 20];
py1=[1 1 1 1 1 1 1 1 1 2.5 3.5 6 3.5 3.5 3.5 6 6 6 8.5 8.5];
pnum=length(px);
pposition22x60=zeros(22,60,pnum);
for i=1:pnum
    pposition22x60(:,:,i)=ppdata22x60(px(i),py1(i),11);
end
pposition(:,:,1)=zeros(22,60);
pposition(:,:,2:21)=pposition22x60;
save('pposition22x60.mat','pposition')
%% pile position data (updated version is ppdata function)
rfdata=importdata(sprintf('C:/Users/Yin/Desktop/Meta/RF_data2/rf%d.txt',i));
cohesion_store = input_image(rfdata,pos);

dist=15;
elev1=2;elev2=12;
column=round(dist/0.5);
row=round(elev1/0.5):round(elev2/0.5);
slope_dist=30;
slope_elev=12;
resolution=0.5;
mould=zeros(slope_elev/resolution,slope_dist/resolution);
mould(row,column)=1;
%% 用mapminmax归一化4D数据
db1=input_database(:,:,1,:);
db1(find(db1==0))=nan; % delete 0, avoiding 0 as the min affects normalization
db_line=reshape(db1,[1,numel(db1)]); % reshape to 1D, numel equal to prod(size(db1))
max_db1=max(db_line);min_db1=min(db_line); % 1D data min and max
db_norm=mapminmax(db_line,0,1); % normalize, there is Nan in matrix
db=reshape(db_norm,size(db1)); % reshape to 4D
db(isnan(db))=0;
input_database(:,:,1,:) = db;
% later I wrote normdata function

%% save some random fields images for gif making
pos=importdata('C:\Users\Yin\Desktop\KL_one\CentroidPosition.txt');
pos=ceil(pos*2);
rfdata=importdata('C:\Users\Yin\Desktop\KL_one\cohesion20.txt');
for ii=1:50
c=rfdata(:,ii);

% scatter(pos(:,1),pos(:,2))

for i=1:length(pos)
    C(pos(i,2),pos(i,1))=c(i);
end

C(find(C==0))=nan;                  % 只关心边坡区域，把剩下为0的地方都删为NAN
C(8,56)=mean([C(8,55),C(8,57)]);
% [F1,ps]=mapminmax(F,0,1);
h=pcolor(C);

% imagesc(C);
set(gca,'YDir','normal');set(gcf,'position',[1000,950,750,250]);
set(gca,'YDir','normal')
set(h, 'EdgeColor', 'none'); % [.6 .6 .6] / 'none' 别漏单引号
xticks([]);yticks([]);
axis off

% saveas(gcf,sprintf('C:/Users/Yin/Desktop/KL_one/pic/%d.png',ii)); 
end
%% Compare the images and direct data input for CNN
c_img1 = imread(sprintf('C:/Users/Yin/Desktop/Meta/images/onelayer/train1/cohesion%d.png',1)); 
pos0=importdata('C:\Users\Yin\Desktop\FLAC_struct\3 firm stratum\CentroidPosition0.txt');
pos=ceil(pos0*2);

d=importdata('C:\Users\Yin\Desktop\FLAC_struct\2 flac3d\cohesion200.txt');
d0=d(:,1);
d1=mapminmax(d0',0,255);

for i=1:length(pos)
    C0(pos(i,2),pos(i,1))=d1(i);
end

C=flipud(C0); % flip the matrix upside down
% xlswrite('C:\Users\Yin\Desktop\image.xlsx',c_img1(:,:,1),'image1') 

%% 画CPT采样点分布图
l=20; % l是CPT竖向有多少个数据点
n=5;
C = gridsamp([5 0.5;25 10], [n l]); % n=5 条采样

scatter(C(:,1),C(:,2),'r','filled');
axis([0 30 0 10]);
set(gcf,'position',[1000,950,750,250]);
xlabel('Distance(m)');ylabel('Elevation(m)');
set(gcf,'color','none');
set(gca,'color','none');
%% 根据几个桩位的FS拟合FS vs position曲线
clc;clear;close all
x=[10 11 12 14 16 18 20];
y0=[1.23047	1.25391	1.25391	1.30078	1.30859	1.33984	1.33984
1.27734	1.29297	1.28516	1.29297	1.30859	1.34766	1.39453
1.24609	1.28516	1.27734	1.30078	0.880859	1.35547	1.35547
1.35547	1.40234	1.39453	1.41016	0.833984	1.47266	1.47266
1.34766	1.38672	1.38672	1.38672	1.40234	1.43359	1.47266
1.14453	1.17578	1.22266	1.33203	1.32422	1.33984	1.33984
1.05859	1.05859	1.05859	1.05078	1.06641	1.10547	1.13672
1.06641	1.09766	1.08984	1.09766	1.05859	1.16016	1.17578
1.44141	1.45703	1.45703	1.48047	1.50391	1.55078	1.55859
1.43359	1.45703	1.44922	1.47266	1.48828	1.54297	1.55859];
y=y0(7,:);
x1=10:0.1:20;
y1=interp1(x,y,x1,'makima');
plot(x1,y1,'c','linewidth',5)
ylabel('Factor of Safety');xlabel('Position(m)');
set(gca,'FontSize',16,'FontWeight','bold' );
grid on

%% Consistency of the CRF
vv=V(:,40);
mu=mean(vv);sigma=std(vv)^2;
X=CRF(:,39);
k=19;
for j=1:length(X)
    for i=1:k-j+1
        diff(i)=(X(i)-mu)*(X(i+j)-mu);
    end
    rho(j)=1/sigma^2/(k-j)*sum(diff);
%     clear diff
end
plot(rho)
%% 画原cohesion图

h=pcolor(X,Y,F);
set(gcf,'position',[1000,800,750,250])     % 图片位置 & 长宽
set(h, 'EdgeColor', 'none');  % [.5 .5 .5] / 'none' 别漏单引号
colorbar('fontsize',12) %EastOutside; %caxis([10,40]);
% xticks([]);yticks([]); box off
% c = colorbar;c.Label.String = 'Cohesion';
% set(h,'linestyle',':','EdgeColor',[.9 .9 .9])
axis off
%% 边坡cohesion property 灰度图
[X,Y]=meshgrid(linspace(0,30,60),linspace(0,10,20));
for i=1:length(rfdata)
    rfn(i)=(rfdata(i)-min(rfdata))/(max(rfdata)-min(rfdata)); % Normalize
end
for i=1:length(pos)
    C(pos(i,2),pos(i,1))=rfn(i); % 转2D
end
C(find(C==0))=nan;                  % 只关心边坡区域，把剩下为0的地方都删为NAN
C(8,56)=mean([C(8,55),C(8,57)]);
% [F1,ps]=mapminmax(F,0,1);
h=pcolor(X,Y,C);set(gca,'YDir','normal');set(gcf,'position',[1000,950,750,250]);
colorbar('fontsize',12);
set(h, 'EdgeColor', 'none'); % [.6 .6 .6] / 'none' 别漏单引号
% xlabel('Length(m)');ylabel('Depth(m)');
xticks([]);yticks([]);
% c = colorbar;c.Label.String = 'Grayscale';
colormap('gray');
axis off
%% 边坡cohesion property 画图
[X,Y]=meshgrid(linspace(0,30,60),linspace(0,10,20));
F(find(F==0))=nan;YX0(find(YX0==0))=nan;
h=pcolor(X,Y,YX0);set(gca,'YDir','normal');set(gcf,'position',[1000,950,750,250]);colorbar EastOutside;caxis([5,38]);
set(h, 'EdgeColor', 'none');
xlabel('Length(m)');ylabel('Depth(m)');
c = colorbar;c.Label.String = 'Cohesioin(kPa)'; % colormap('jet');
%% 看CRF最大最小值，为生成图像灰度缩放确定范围
load('C:\Users\Yin\Desktop\CRF1\3cpt\ForFOS200\allCRF.mat');
max_crf(1:200,1) = max(all,[],[1 2]);
plot(max_crf);
all(find(all==0))=99;
min_crf(1:200,1) = min(all,[],[1 2]);
figure(2);plot(min_crf);
%% 
clc;clear;close all
data1 = load('C:\Users\Yin\Desktop\CRF1\5cpt\2000\cohesion5cpt2000.txt');
data2 = load('C:\Users\Yin\Desktop\CRF1\5cpt\ForFOS200\cohesion.txt');

data0=data1(:,1:100);
data = [data0 data2];
save(['C:\Users\Yin\Desktop\CRF1\5cpt\ForFOS200\','cohesion1+1.txt'],'data','-ascii')
% save(['C:\Users\Yin\Desktop\CRF1\5cpt\100t\','cohesion5cpt100t.txt'],'data12','-ascii')
%% 看一些URF图
clc;clear ; close all
rfdata=load('C:\Users\Yin\Desktop\cnn1L\2 flac3d\1 URF\100\cohesion.txt');
pos=importdata('C:\Users\Yin\Desktop\CPT_KL_CRF\CentroidPosition.txt'); % coordinate910
pos=ceil(pos*2);
for j=1:size(rfdata,2)
    for i=1:length(pos)
        F(pos(i,2),pos(i,1),j)=rfdata(i,j); % 转2D
    end
end
for k=1:30
    start = 50;
    subplot(6,5,k)
    imagesc(F(:,:,k+start-1));set(gca,'YDir','normal');
    title(sprintf('%d th',k+start-1))
end

%% 一个大txt拆分
clc;clear;close all
data = load('C:\Users\Yin\Desktop\CRF1\5cpt\cohesion1+1.txt');
for i =1:size(data,2)
    filename = sprintf('rf%d.txt',i);
    data1 = data(:,i);
    save(['C:\Users\Yin\Desktop\cnn1L\1 imagenerator\RF_data\',filename],'data1','-ascii')   
end
%% 合并多个RF到一个txt
clc;clear;close all
nn=1000;
for i=1:nn
    filename = sprintf('C:/Users/Yin/Desktop/15m_slope_2layer/指数相关函数小概率/cfdata/numS%d.txt',i);
    datai=load(filename);
    rf_c(:,i)=datai(:,1);
    rf_phi(:,i)=datai(:,2);
end
% save(['D:\202210_surrogate_pile_RBDO\FLAC_struct2\KL-RF\pf_c.txt'],'rf_c','-ascii')
% save(['D:\202210_surrogate_pile_RBDO\FLAC_struct2\KL-RF\pf_phi.txt'],'rf_phi','-ascii')

%% 看生成随机场的PDF
data=load('C:\Users\Yin\Desktop\cnn1L\2 flac3d\1 URF\100\cohesion.txt');
for i=1:36
    subplot(6,6,i)
    hist(data(:,i),5:60)
    title(sprintf('%d',i))
end


%% CNN surrogate performance
Net= 5 ;

abab=eval(['FS_Net',num2str(Net)]);
scatter(FS_ture,abab,'r','marker','^'); % 测试是红色
set(gca,'fontsize',12); % ,'linewidth',4,'fontname','Times'
hold on;
% xlabel('true','fontsize',fsize);ylabel('predicted','fontsize',fsize);
grid on; % box on
% 1：1图
lb = min([FS_ture;abab])-0.4;
ub = max([FS_ture;abab])+0.2;
range=lb:0.01:ub;
fsize=16;
axis([lb ub,lb ub])

% train和validate对比
valInd = (indices == 2); % 第 N 折
trainInd = ~valInd;
train_input = input_database(:,:,:,trainInd);
train_output = output_database(trainInd);
val_input = input_database(:,:,:,valInd);
val_output = output_database(valInd);

for i = 1:length(find(trainInd))
    cv_train(i,1) = predict(eval(['Net',num2str(Net)]),train_input(:,:,:,i));
end
for i = 1:length(find(valInd))
    cv_validate(i,1) = predict(eval(['Net',num2str(Net)]),val_input(:,:,:,i));
end

scatter(train_output,cv_train,'k'); % 训练是黑色
hold on;
scatter(val_output,cv_validate,'b'); % 验证是蓝色
hold on;
plot(range,range,'k');  % 1:1 Line
xlabel('FOS by FDM');ylabel('FOS by CNN');
legend('test','train','validation','1:1','location','southeast','fontsize',fsize); %
set(gca,'FontSize',fsize);
legend('boxoff');
grid on
box on
axis([lb ub,lb ub])

% text(0.83,1.32,{'(c)   R^2 = 0.9995','       RMSE = 0.0269'},'fontsize',fsize);
% calculate R2
Y = eval(['FS_Net',num2str(Net)]);
T = FS_ture;
R2=1-sum(power(Y-T,2))/sum(power(T-mean(T),2))

RMSE = mean([info1.FinalValidationRMSE,info2.FinalValidationRMSE,info3.FinalValidationRMSE,...
    info4.FinalValidationRMSE,info5.FinalValidationRMSE])

% 折线图
% figure
% plot(Z_tureFoS);
% hold on;plot(ZFoS_ConvNet5)
% xlabel('nth FoS');ylabel('value');

%% 一次训练的Loss和RMSE
net=5;

xa=1:length(eval(['info',num2str(net),'.TrainingLoss']));

yyaxis left
semilogy(xa,eval(['info',num2str(net),'.TrainingLoss'])); % 训练集MAE
set(gca,'fontsize',12); % ,'linewidth',4,'fontname','Times'
xlabel('Number of training epochs','fontsize',14);
ylabel('Loss (MAE) of FOS','fontsize',14);

yyaxis right
semilogy(xa,eval(['info',num2str(net),'.TrainingRMSE'])); % 训练集RMSE
ylabel('RMSE of FOS','fontsize',14);
hold on
semilogy(xa,eval(['info',num2str(net),'.ValidationRMSE'])); % 验证集RMSE

yyaxis left
semilogy(xa,eval(['info',num2str(net),'.ValidationLoss'])); % 验证集MAE
legend('Training Loss','Validation Loss','Training RMSE','Validation RMSE','fontsize',12)
set(gcf,'position',[1000,800,800,300])
grid on
text(1800,0.07,{'Early stop at 2806 epochs'},'fontsize',14);
%% CNN训练预测结果做表格

fs2000=[FoS_ConvNet1_mean FoS_ConvNet2_mean FoS_ConvNet3_mean FoS_ConvNet4_mean FoS_ConvNet5_mean FoS_ConvNet_5CV_mean];
std2000=[FoS_ConvNet1_std FoS_ConvNet2_std FoS_ConvNet3_std FoS_ConvNet4_std FoS_ConvNet5_std FoS_ConvNet_5CV_std];
Pf2000=[Pf_ConvNet1 Pf_ConvNet2 Pf_ConvNet3 Pf_ConvNet4 Pf_ConvNet5 Pf_ConvNet_5CV];
beta=RI_ConvNet_5CV
mean([fs2000' std2000' Pf2000' ])
%% 看这个插值结果用flac算fos是多少，这个是保存个用来算的txt
for i=1:length(pos)
    interp(i,1)=YX0(pos(i,2),pos(i,1));
end
save(['C:\Users\Yin\Desktop\cnn1L\2 flac3d\cohesion.txt'],'interp','-ascii')

%% 取出一个cohesion并保存
clc;clear ; close all
data=load('C:\Users\Yin\Desktop\cnn1L\2 flac3d\1 URF\100\cohesion.txt');
example1=data(:,50); % 第50
save(['C:\Users\Yin\Desktop\cnn1L\2 flac3d\cohesion.txt'],'example1','-ascii')

%% 用生成的矩形CRF数据，得到边坡CRF，并存成txt
clc;clear ; close all
load('C:\Users\Yin\Desktop\CRF1\allCRF.mat');
C=load('C:\Users\Yin\Desktop\cnn1L\3 KL\CentroidPosition.txt');
C=ceil(C*2);
for j=1:size(all,3)
    tempF=all(:,:,j);
    for i=1:length(C)
        crf(i,j)=tempF(C(i,2),C(i,1));
    end
    crf(find(crf<0))=0.1;
end
negv=crf(find(crf<0));plot(negv);
neg=find(crf<0);
negind=unique(ceil(neg/910))

save(['C:\Users\Yin\Desktop\CRF1\cohesion1cpt2000.txt'],'crf','-ascii')

%% 功能同上，这个是CRF的base
clc;clear ; close all
% rfdata是CRF的base矩形RF，形式为1200*1
rfdata=importdata('C:\Users\Yin\Desktop\CPT_KL_CRF\cohesion1.txt');
% pos是边坡坐标
pos=load('C:\Users\Yin\Desktop\cnn1L\3 KL\CentroidPosition.txt');
pos=ceil(pos*2);

for i=1:length(pos)
    F(pos(i,2),pos(i,1))=rfdata(i); % F是边坡RF，二维
end

for i=1:length(pos)
    crf(i,:)=F(pos(i,2),pos(i,1)); % 保存成1D数据，用来导入FLAC
end

imagesc(F)
set(gca,'YDir','normal')
save(['C:\Users\Yin\Desktop\cnn1L\2 flac3d\cohesionb.txt'],'crf','-ascii')
%% 生成散点坐标，间隔0.5（KL-RF的画图用）
clc;clear;close all
x=0.25:0.5:29.75;
y=0.25:0.5:9.75;
[Y,X]=meshgrid(x,y);
k=0;
for i=1:length(y)
    for j=1:length(x)
        k=k+1;
        pos(k,:)=[Y(i,j) X(i,j)]; % 生成两列的坐标
    end
end
% num1=length(x)*length(y)*8/15;num2=num1/8*7; % 分两层
% ind2=pos(1:num1,:);ind1=pos(num1+1:end,:);
% save(['C:\Users\Yin\Desktop\cnn\KL\cen_xz_lay1.txt'],'ind1','-ascii')
% save(['C:\Users\Yin\Desktop\cnn\KL\cen_xz_lay2.txt'],'ind2','-ascii')
% save(['C:\Users\Yin\Desktop\CPT_KL_CRF\CentroidPosition.txt'],'pos','-ascii')
scatter(pos(:,1),pos(:,2))
%% 1D坐标转2D（KL-RF的画图用）1层边坡
clc;clear;close all

pos=importdata('D:\202204_CRF-CNN_ConfPaper\cnn1L\3 KL\CentroidPosition.txt');
pos=ceil(pos*2);
% rfdata=importdata('D:\202210_surrogate_pile_RBDO\FLAC_struct2\RSM\KL_SRM\try3_04306.txt');
c=rfdata(:,1);

% scatter(pos(:,1),pos(:,2))

for i=1:length(pos)
    C(pos(i,2),pos(i,1))=c(i);
end

imagesc(C)
set(gca,'YDir','normal')


% 查看坐标排列顺序
% for i=1:length(pos)
%     scatter(pos(i,1),pos(i,2));
%     pause(1/10000)
%     hold on
% end

%% 1D坐标转2D（KL-RF的画图用）2层边坡
clc;clear;close all

pos1=importdata('C:\Users\Yin\Desktop\15m_slope_2layer\cnn\flac2\cen_xz_lay1.txt');
pos1=ceil(pos1*2);
pos2=importdata('C:\Users\Yin\Desktop\15m_slope_2layer\cnn\flac2\cen_xz_lay2.txt');
pos2=ceil(pos2*2);
rfdata=importdata('C:\Users\Yin\Desktop\15m_slope_2layer\指数相关函数\cfdata\numS2.txt');
c=rfdata(:,1);
f=rfdata(:,2);

L1C=c(1:length(pos1));
L2C=c(length(pos1)+1:length(pos1)+length(pos2));
L1F=f(1:length(pos1));
L2F=f(length(pos1)+1:length(pos1)+length(pos2));

scatter(pos1(:,1),pos1(:,2))
hold on
scatter(pos2(:,1),pos2(:,2))

for i=1:length(pos1)
    C(pos1(i,2),pos1(i,1))=L1C(i);
end
for i=1:length(pos2)
    C(pos2(i,2),pos2(i,1))=L2C(i);
end
% imagesc(C)
% set(gca,'YDir','normal')

for i=1:length(pos1)
    F(pos1(i,2),pos1(i,1))=L1F(i);
end
for i=1:length(pos2)
    F(pos2(i,2),pos2(i,1))=L2F(i);
end
% figure
% imagesc(F)
% set(gca,'YDir','normal')

% 查看坐标排列顺序
% for i=1:length(pos1)
%     scatter(pos1(i,1),pos1(i,2));
%     pause(1/1000)
%     hold on
% end


%% kriging DACE Toolbox
%% Try Kriging
clc;clear;close all
load('C:\Users\Yin\Downloads\kriging_toolbox\dace\data1.mat');
% 模型参数设置，无特殊情况不需修改，见说明书
theta = [10 10]; lob = [1e-1 1e-1]; upb = [20 20];

[dmodel, perf] = dacefit(S, Y,@regpoly2, @corrspline, theta, lob, upb);

%创建一个40*40的格网，标注范围为0-100，即格网间距为2.5
X = gridsamp([0 0;100 100], 40);

% X=[83.731	32.36];     %单点预测的实现
%格网点的预测值返回在矩阵YX中，预测点的均方根误差返回在矩阵MSE中
[YX,MSE] = predictor(X, dmodel);

X1 = reshape(X(:,1),40,40);
X2 = reshape(X(:,2),40,40);
YX = reshape(YX, size(X1));         %size(X1)=40*40

figure(1)
surf(X1, X2, YX)         %绘制预测表面，也可以用mesh
hold on
plot3(S(:,1),S(:,2),Y,'.k', 'MarkerSize',10)    %绘制原始散点数据
hold off
shading interp % 不需要平滑就注掉

figure(2)
mesh(X1, X2, reshape(MSE,size(X1)));  %绘制每个点的插值误差大小
% shading interp

%% 查看DACE工具包不同 回归函数 和 相关函数 组合误差
clc;clear;close all
load('C:\Users\Yin\Downloads\kriging_toolbox\dace\data1.mat');
%模型参数设置
theta = [10 10]; lob = [1e-1 1e-1]; upb = [20 20];
X = gridsamp([0 0;100 100], 40);

reg={@regpoly0 @regpoly1 @regpoly2};
cor={@correxp @corrgauss  @corrlin @corrspherical @corrspline @correxpg};

% 列出不同 回归函数 和 相关函数 组合
for j=1:6
    for i=1:3
        a{i}=reg{i};b{j}= cor{j};
    end
end
% 计算不同组合的预测结果、误差
for j=1:5  %第三个函数报错，放在最后了没调用
    for i=1:3
        [dmodel, perf] = dacefit(S, Y,reg{i}, cor{j}, theta, lob, upb);
        [YX,MSE] = predictor(X, dmodel);
        err(i,j)=mean(MSE);
    end
end

% X1 = reshape(X(:,1),40,40);
% X2 = reshape(X(:,2),40,40);
% YX = reshape(YX, size(X1));         %size(X1)=40*40

plot(err(:,:));xlabel('3个回归函数');ylabel('MSE');
figure
plot(err');xlabel('5个相关函数');ylabel('MSE');

% 试完最好的是 3 reg + 5 cor

%% 生成随机点
% % figure;
% % set(gcf,'Position',[400,300,200,200]);
% a1 = 40;
% % set(axes,'Color',[128/255 128/255 128/255]);
% % hold on
% C = randi([1,129],a1,1);   % 随机生成整数型函数
% C1 = randi([1,129],a1,1);
%
% plot(C,C1,'o','Color','r','MarkerfaceColor','r','MarkerSize',4);
% % hold on
% % set(gca,'xtick',[],'xticklabel',[]); % 去坐标
% % hold on
% % set(gcf,'Color',[128/255 128/255 128/255]);
%
% for i=1:40
%     v(i)=F(C1(i),C(i));
% end

%% 自己随便弄的对角相关RF

nx=500;
ny=500;

a=zeros(ny,nx);
a(1,1)=rand(1);
for i=2:nx
    a(1,i)=a(1,i-1)+randn(1)*0.02;
end
for j=2:ny
    a(j,1)=a(j-1,1)+randn(1)*0.02;
end
for j=2:ny
    for i=2:nx
        a(j,i)=(a(j,i-1)+a(j-1,i))/2+randn(1)*0.02;
    end
end

figure
clims = [-1.5,2.3];
gca=imagesc(real(a));
% axis xy                  % axis ij : 反向，y轴的值从上往下递增
colormap(jet)
colorbar EastOutside


%% 看一下相关函数的形态

tau_H=0.01:0.1:10;
tau_V=0.01:0.1:10;
[X,Y]=meshgrid(tau_H,tau_V);
theta_h=10;
theta_v=2; % 两个方向的相关距离
rho=zeros(length(tau_H),length(tau_V));
for i=1:length(tau_H)
    for j=1:length(tau_V)
        tau_h=tau_H(i);
        tau_v=tau_V(j);
        rho(i,j)=exp(-tau_h/theta_h-tau_v/theta_v);        % Li Yajun 3D modefied 2D
%         rho(i,j)=exp(-2*abs(tau_h)/theta_h-sqrt((2*tau_v/theta_v)^2));        % Li Yajun 3D modefied 2D
%         rho(i,j)=exp(-pi*tau_h^2/(theta_h^2)-pi*tau_v^2/(theta_v^2));       % (2020 Gong) squared exponential
%         rho(i,j)=exp(-tau_h/theta_h)*exp(-tau_v/theta_v);                   % (2013 Huang) exponential covariance
%         rho(i,j)=exp(-2*sqrt(power((tau_h/theta_h),2)+power((tau_v/theta_v),2)));  % (2011 Masin) Markov
    end
end
figure
mesh(X,Y,rho);
surf(X,Y,rho);
shading interp
xlabel('x');ylabel('y');zlabel('\rho');
%基本上可以得到结论：1）都是一个形状，两边逐渐递减，但减的速度有明显不同；2）相关长度不一样两边就不对称

% Correlation='Markov';
% theta=5;
% tau=0.01:0.1:10;
% for i=1:length(tau)
%     if strcmp(Correlation,'Gaussian')==1                  % Gaussian
%         rho(i)=exp(-pi*(abs(tau(i))/theta)^2);
%     elseif strcmp(Correlation,'Markov')==1                % Markov
%         rho(i)=exp(-2*abs(tau(i))/theta);
%     elseif strcmp(Correlation,'Triangular')==1            % Triangular
%         if abs(tau(i))<=theta
%             rho(i)=1-abs(tau(i))/theta;
%         else
%             rho(i)=0;
%         end
%     elseif strcmp(Correlation,'Spherical')==1             % Spherical
%         if abs(tau(i))<=theta
%             rho(i)=1-1.5*abs(tau(i)/theta)+0.5*abs(tau(i)/theta)^3;
%         else
%             rho(i)=0;
%         end
%     end
% end
% plot(rho)


%% DWT
% n=64;
% g=eye(n);
% W=zeros(n);
% qmf = MakeONFilter('Haar');
% for i=1:n
%     W(:,i)=FWT_PO(g(:,i),1,qmf);
% end
% mesh(W)
%
%
% n=4;
% g=[4.1;4.3;4.5;4.7];
% W=zeros(n);
% qmf = MakeONFilter('Haar');
% for i=1:n
%     W(:,i)=FWT_PO(g(:,i),1,qmf);
% end
% mesh(W)

% n=4;
% g=[4.1;4.3;4.5;4.7];
% w=zeros(n);
% qmf = MakeONFilter('Haar');
% w=FWT_PO(g,1,qmf)


% [cA,cD] = dwt(g,'Haar');
% w=[cA;cD]


%% gamma function

fplot(@gamma)

%% gamma分布

x=0:0.1:20;
y=gampdf(x,3,5);
plot(x,y)

%% 指数分布

x=0:0.5:50;
y=exppdf(x,3);
plot(x,y)

%% kai方分布

a=randn(50000,1);
% histogram(a,'Normalization','count')

b=normrnd(1,0.5,[50000,1]);
% histogram(b,'Normalization','pdf')

c=a.*a+b.*b; % 自由度为2
histogram(c,'Normalization','pdf')

%% 二项分布

x=0:50;
y=binopdf(x,500,0.05);
figure
bar(x,y); %柱状图
xlabel('x/变量','fontsize',15);
ylabel('p/概率','fontsize',15);
title('二项分布 X~b(500,0.05)','fontsize',15);

%某人向空中抛硬币100次，落下是正面向上的概率0.5,100次中正面向上的次数记为X
%(1) 求X=45的概率，(2) 绘制分布列图象。。
% 计算x=45的概率
px=binopdf(45,100,0.5)
%作图
x=1:100;
figure
p=binopdf(x,100,0.5);
plot(x,p);title('概率分布图')

%% Poisson分布

x=0:50;
y=poisspdf(x,5);
bar(x,y)
xlabel('x/变量','fontsize',15); ylabel('y/概率','fontsize',15);
title('poisson分布 x~π（2）','fontsize',15);

%% 正态分布

x=-15:0.01:15;
y=pdf('norm',x,0,1);
plot(x,y);
xlabel('x/变量','fontsize',15);ylabel('y/概率','fontsize',15);
title('正态分布 x~N(0,1)','fontsize',15);

%% debug
try
    abcde
    disp('OK')
catch
    disp('wrong')
end

