folder = 'C:\Users\Duygu\Desktop\Veri Madenciliği\parkinson';
fullFileName = fullfile(folder, 'parkinsons.csv')

parkinsonData = readtable(fullFileName);
data= parkinsonData(:,[2:24]);

%30 u test olarak kullanıldı.
pt = cvpartition(data.status,"HoldOut",0.2);
X= removevars(data,'status')
Y=data.status;

pdTrain = training(pt);
pdTest = test(pt);

XTrain = X(pdTrain,:);
YTrain = Y(pdTrain);
XTest = X(pdTest,:);
YTest = Y(pdTest);

svmModel = fitcsvm(XTrain,YTrain);

%Model oluşturmak için oluşan kayıp 
trainError = resubLoss(svmModel);

cvMdl = crossval(svmModel);

%cross validation hesaplaması sonucu CVMdl tarafından elde edilen ortalama sınıflandırma hatası.

cvtrainError = kfoldLoss(cvMdl);

cvtrainAccuracy = 1-cvtrainError;


%model ve test sonuçları karşılaştırılarak elde edilen hata
newError = loss(svmModel,XTest, YTest);

newAccuracy = 1-newError;


pred2 = predict(svmModel,XTest);
%gerçek etiketlerden ve tahmin edilen etiketlerden bir karışıklık matrisi grafiği oluşturur 
%Karışıklık matrisinin satırları gerçek sınıfa karşılık gelir ve sütunlar tahmin edilen sınıfa karşılık gelir.
%Köşegen ve köşegen dışı hücreler, sırasıyla doğru ve yanlış sınıflandırılmış gözlemlere karşılık gelir.
figure
confusionchart(pred2, YTest);

confmat = confusionmat(YTest,pred2);

confmatT= confmat'
diagonal = diag(confmatT)

%Kesinlik (Precision) ise Positive olarak tahminlediğimiz değerlerin 
%gerçekten kaç adedinin Positive olduğunu göstermektedir. Mean fonk ile
%ortalamsı alındı.

sum_of_rows = sum(confmatT, 2)
precision = diagonal ./ sum_of_rows
overall_precision = mean(precision)

%Duyarlılık (Recall) ise Positive olarak tahmin etmemiz gereken işlemlerin ne kadarını
%Positive olarak tahmin ettiğimizi gösteren bir metriktir.
sum_of_columns=sum(confmatT,1)
recall = diagonal ./ sum_of_columns'
overall_recall = mean (recall)


%F1 Score değeri bize Kesinlik (Precision) ve Duyarlılık (Recall) değerlerinin harmonik ortalamasını göstermektedir.

f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision + overall_recall))

figure
[labels,scores]=resubPredict(svmModel)
[s1,m1,T,AUC] = perfcurve(svmModel.Y,scores(:,2),'1')%1 positif değeri isteidği için
plot(s1,m1,'LineWidth',3)

AUC

CVMdl = crossval(svmModel);
kloss = kfoldLoss(CVMdl)
