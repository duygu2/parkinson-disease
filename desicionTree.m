folder = 'C:\Users\Duygu\Desktop\Veri Madenciliği\parkinson';
fullFileName = fullfile(folder, 'parkinsons.csv')

parkinsonData = readtable(fullFileName);
data= parkinsonData(:,[2:24]);
pt = cvpartition(data.status,"HoldOut",0.3);

pdTrain = data(training(pt),:);
pdTest = data(test(pt),:);

%özellikler
meas= removevars(pdTrain,'status');

%sınıflandırma değerleri
species=pdTrain.status;

model= fitctree(meas,species);

figure
view(model, 'mode', 'graph');

tahmin = predict(model,pdTest);

confusionchart(tahmin, pdTest.status);

cvc=crossval(model);
kloss = kfoldLoss(cvc)
cvcAccurary = 1- kloss


ctree = fitctree(meas,species,Crossval="on");

%gerçek etiketlerden ve tahmin edilen etiketlerden bir karışıklık matrisi grafiği oluşturur 
%Karışıklık matrisinin satırları gerçek sınıfa karşılık gelir ve sütunlar tahmin edilen sınıfa karşılık gelir.
%Köşegen ve köşegen dışı hücreler, sırasıyla doğru ve yanlış sınıflandırılmış gözlemlere karşılık gelir
confmat = confusionmat(pdTest.status,tahmin);
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

% ./ Elemanter bölme

sum_of_columns=sum(confmatT,1)
recall = diagonal ./ sum_of_columns'
overall_recall = mean (recall)

%F1 Score değeri bize Kesinlik (Precision) ve Duyarlılık (Recall) değerlerinin harmonik ortalamasını göstermektedir.
f1_score = 2* ((overall_precision*overall_recall)/ (overall_precision + overall_recall))

figure
[labels,scores]=resubPredict(model)
[s1,m1,T,AUC] = perfcurve(model.Y,scores(:,2),'1')%1 positif değeri isteidği için
plot(s1,m1,'LineWidth',3)



                                                                                                                                           