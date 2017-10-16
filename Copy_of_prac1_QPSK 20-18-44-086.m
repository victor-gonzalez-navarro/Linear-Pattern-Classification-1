% Updated to Matlab2015
% MC feb15

clear
close all

%OPTIONS AND DESIGN PARAMETERS:
i_scatter=1;
% SIGNAL TO NOISE RATIO (dB)
SNR=input('SNR(dB)=');
% Inter-Symbol Distance
dist=1;
n_classes=4;                     %QPSK
n_samples=[1000;1000;1000;1000];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ro= 0.0*ones(1,n_classes);	   %Correlation between gaussian components
ro=[0.5 0 -0.5 0.8];           %To be activated in Part 2, second part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_feat=2;
M_Means=0.5*dist*[1,1;1,-1;-1,1;-1,-1]; %QPSK mean vector;
clear dist
% Energy computation
Energy=0;
for i_class=1:n_classes
   V=squeeze(M_Means(i_class,:));
   Energy=Energy+V*V';
end
Energy=Energy/n_classes;
%Variance computation
SNR=10^(SNR/10);
sigma=Energy/SNR;
clear V Energy
%Covariance matrix
M_covar=zeros(n_feat,n_feat,n_classes);
sigma=ones(1,n_classes)*sigma/n_feat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sigma(1) = 30;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i_class=1:n_classes
   M_covar(:,:,i_class)=sigma(i_class)*[1 ro(i_class);ro(i_class) 1];	%Covariance Matrix class i_class
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_class=1:n_classes
    Autoval = eig(squeeze(M_covar(:,:,i_class)))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear sigma

%% Dataset generation
X=[];
Labels=[];
for i_class=1:n_classes
    X=[X;mvnrnd(M_Means(i_class,:),M_covar(:,:,i_class),n_samples(i_class))];
    Labels=[Labels; i_class*ones(n_samples(i_class),1)];
end
clear i_class

%% Create a default (linear) discriminant analysis classifier:
linclass = fitcdiscr(X,Labels)
Linear_out = predict(linclass,X);
Linear_Pe=sum(Labels ~= Linear_out)/length(Labels);
fprintf(1,' error Linear = %g   \n', Linear_Pe)

%% Create a quadratic discriminant analysis classifier:
quaclass = fitcdiscr(X,Labels,'discrimType','quadratic')
Quadratic_out = predict(quaclass,X);
Quadratic_Pe=sum(Labels ~= Quadratic_out)/length(Labels);
fprintf(1,' error Quadratic = %g   \n', Quadratic_Pe)

%% create a scatter plot of the data
V=randperm(length(Labels));
gscatter(X(V,1),X(V,2),Labels(V),'krbg','ov^*',[],'off');
grid
hold on
Xmin=min(X(:,1));
Xmax=max(X(:,1));
Ymin=min(X(:,2));
Ymax=max(X(:,2));

%% Plot the LINEAR classification boundaries for the class 1.
for i_class=2:4
    K = linclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the linear
    L = linclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
    % Plot the curve K + [x,y]*L  = 0.
    f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
    h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
    set(h,'Color','k','LineWidth',2);
end

%% Plot the QUADRATIC classification boundaries for the class 1.
for i_class=2:4
    K = quaclass.Coeffs(1,i_class).Const; % retrieve the coefficients for the quadratic
    L = quaclass.Coeffs(1,i_class).Linear;% boundary between the first and second classes
    Q = quaclass.Coeffs(1,i_class).Quadratic;
    % Plot the curve K + [x1,x2]*L + [x1,x2]*Q*[x1,x2]' = 0.
    f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
        (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
    h = ezplot(f,[Xmin,Xmax,Ymin,Ymax]);
    set(h,'Color','r','LineWidth',2);
end
title('LC Boundaries - black, QC - red')
clear Xmin Xmax Ymin Ymax h K L Q

%% Confusion matrices
CM_Lineal=confusionmat(Labels,Linear_out)
CM_quadratic=confusionmat(Labels,Quadratic_out)


