folder = 'C:\Users\Duygu\Desktop\Veri Madenciliği\parkinson';
fullFileName = fullfile(folder, 'parkinsons.csv')

parkinsonData = readtable(fullFileName);
data= parkinsonData(:,[2:24]);

X= removevars(data,'status')
Y=data.status;


%30 u test olarak kullanıldı.
pt = cvpartition(data.status,"HoldOut",0.3);

pdTrain = training(pt);
pdTest = test(pt);

XTrain = X(pdTrain,:);
YTrain = Y(pdTrain);
XTest = X(pdTest,:);
YTest = Y(pdTest);

Model= fitcnb(XTrain, YTrain);

pred2 = predict(Model,XTest);

%gerçek etiketlerden ve tahmin edilen etiketlerden bir karışıklık matrisi grafiği oluşturur 
%Karışıklık matrisinin satırları gerçek sınıfa karşılık gelir ve sütunlar tahmin edilen sınıfa karşılık gelir.
%Köşegen ve köşegen dışı hücreler, sırasıyla doğru ve yanlış sınıflandırılmış gözlemlere karşılık gelir.
confusionchart(pred2, YTest);

confmat = confusionmat(YTest,pred2);
confmatT= confmat';
diagonal = diag(confmatT);

%Kesinlik (Precision) ise Positive olarak tahminlediğimiz değerlerin 
%gerçekten kaç adedinin Positive olduğunu göstermektedir. Mean fonk ile
%ortalamsı alındı.
    
sum_of_rows = sum(confmatT, 2);
precision = diagonal ./ sum_of_rows;
overall_precision = mean(precision);

%Duyarlılık (Recall) ise Positive olarak tahmin etmemiz gereken işlemlerin ne kadarını
%Positive olarak tahmin ettiğimizi gösteren bir metriktir.

% ./ Elemanter bölme
sum_of_columns=sum(confmatT,1);
recall = diagonal ./ sum_of_columns';
overall_recall = mean (recall);

%F1 Score değeri bize Kesinlik (Precision) ve Duyarlılık (Recall) değerlerinin harmonik ortalamasını göstermektedir.


f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision + overall_recall));

figure
[labels,scores]=resubPredict(Model);
[s1,m1,T,AUC] = perfcurve(Model.Y,scores(:,2),'1');%1 positif değeri isteidği için
plot(s1,m1,'LineWidth',3);

cvMdl = crossval(Model); 

%cross validation hesaplaması sonucu CVMdl tarafından elde edilen ortalama sınıflandırma hatası.

cvtrainError = kfoldLoss(cvMdl);
disp("Cross validation hatası:  " + cvtrainError);

%cross validation eğitim doğruluğu

cvtrainAccuracy = 1-cvtrainError;
disp("Cross validation doğruluğu:  " + cvtrainAccuracy);
%model ve test sonuçları karşılaştırılarak elde edilen hata

newError = loss(Model,XTest, YTest);
disp("Sınıflandırma hatası:  " + newError);

newAccuracy = 1-newError;
disp("Sınıflandırma doğruluğu:  " +newAccuracy);

AUC