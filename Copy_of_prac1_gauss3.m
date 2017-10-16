% Updated to Matlab2015
% when a 3D classifier is computed, by the moment it is not plotted
% MT,JV marzo 2013
% MC sept13, febrero 2016

%% OPTIONS
clear
close all
i_hi=1;                         %0 NO /1 YES: HISTOGRAM FUNCTIONS
i_scplot=1;					    %0 NO /1 YES: scatterplot of features
i_roc=1;						%0 NO /1 YES: ROC computation

%% Parameter initialitation
% SIGNAL TO NOISE RATIO (dB) AND INTER-CLASSES DISTANCE
disp(' ')
SNR=input('SNR (dB) = ');
dist=1;  % Distance between classes mean ;
n_classes=2;
n_samples=[1000;1000];
n_feat=3;
M_Means=0.5*dist*[1,1,1;-1,-1,-1]/sqrt(n_feat); 		%Matrix containing two  Mean vector

% Energy computation
Energy=0;
for i_classes=1:n_classes
   V=squeeze(M_Means(i_classes,:));
   Energy=Energy+V*V';
end
Energy=Energy/n_classes;

%noise variance computation
SNR=10^(SNR/10);
sig=Energy/SNR;
sig=sig/n_feat;
clear V Energy

%Covariance matrix
M_covar=zeros(n_feat,n_feat,n_classes);
sigma=sig*[1 1];
clear sig
for i_clase=1:n_classes
   M_covar(:,:,i_clase)=sigma(i_clase)*eye(n_feat);	%Covariance Matrix clase i_clase
end

%% Dataset generation
X=[];
Labels=[];
for i_class=1:n_classes
    X=[X;mvnrnd(M_Means(i_class,:),M_covar(:,:,i_class),n_samples(i_class))];
    Labels=[Labels; (i_class-1)*ones(n_samples(i_class),1)];
end
clear i_class

%%  HISTOGRAMS
if i_hi==1
    figure('name','HISTOGRAMS')
    index= Labels==0;
    for i_feat=1:n_feat
        subplot(3,2,2*i_feat-1)
        histfit(X(index,i_feat))
        grid
        zoom on
        ylabel(['Feat ',num2str(i_feat)]);
        title('Class 0');
    end
    index= Labels==1;
    for i_feat=1:n_feat
        subplot(3,2,2*i_feat)
        histfit(X(index,i_feat))
        grid
        zoom on
        ylabel(['Feat ',num2str(i_feat)]);
        title('Class 1')
    end
    clear index i_feat
end

%% SCATTER PLOT 
if i_scplot==1
    varNames = {'feat 1' 'feat 2' 'feat 3'};
    figure('name','Scatter Plot')
    V=randperm(length(Labels));
    gplotmatrix(X(V,:),X(V,:),Labels(V),'br','.',[],'on','hist',varNames,varNames)
    grid
    zoom on
    % Plot en 3D
    figure('name','Plot 3D clusters')
    index=find(Labels==1);
    plot3(X(index,1),X(index,2),X(index,3),'b+');
    hold on
    index=find(Labels==0);
    plot3(X(index,1),X(index,2),X(index,3),'r*');
    grid
    clear index V varNames
end

%% Create a default (linear) discriminant analysis classifier:
linclass = fitcdiscr(X,Labels)
[Linear_out, Score_linear]= predict(linclass,X);
Linear_Pe=sum(Labels ~= Linear_out)/length(Labels);
fprintf(1,' error Linear = %g   \n', Linear_Pe)

%% Create a quadratic discriminant analysis classifier:
quaclass = fitcdiscr(X,Labels,'discrimType','quadratic')
[Quadratic_out, Score_qua]= predict(quaclass,X);
Quadratic_Pe=sum(Labels ~= Quadratic_out)/length(Labels);
fprintf(1,' error Quadratic = %g   \n', Quadratic_Pe)
%% ROC & CONFUSION MATRIX
if i_roc==1
    figure
    plotroc(Labels',flip(Score_linear'),'Linear',Labels',flip(Score_qua'),'Quadratic');
    CM_Lineal=confusionmat(Labels,Linear_out)
    CM_quadratic=confusionmat(Labels,Quadratic_out)
end
%% Quadratic Mahalanobis distance
% d(I) = (Y(I,:)-mu)*inv(SIGMA)*(Y(I,:)-mu)'
% d = mahal(Y,X)
Ind_class0= Labels==0;
Data_class0=X(Ind_class0,:);
Ind_class1= Labels==1;
Data_class1=X(Ind_class1,:);
dist=mahal(Data_class0,Data_class1);
d_01=mean(dist)
dist=mahal(Data_class1,Data_class0);
d_10=mean(dist)
clear Ind_class1 Data_class1 Ind_class0 Data_class0 dist