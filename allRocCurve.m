folder = 'C:\Users\Duygu\Desktop\Veri Madenciliği\parkinson';
fullFileName = fullfile(folder, 'parkinsons.csv')

parkinsonData = readtable(fullFileName);
data= parkinsonData(:,[2:24]);

%30 u test olarak kullanıldı.
pt = cvpartition(data.status,"HoldOut",0.3);

X= removevars(data,'status')
Y=data.status;

pdTrain = training(pt);
pdTest = test(pt);

XTrain = X(pdTrain,:);
YTrain = Y(pdTrain);
XTest = X(pdTest,:);
YTest = Y(pdTest);

pdTrainKNN = data(training(pt),:);
nbModel= fitcnb(XTrain, YTrain);
knn_model = fitcknn(XTrain, YTrain,'NumNeighbors',5);
svmModel = fitcsvm(XTrain,YTrain);
dtModel= fitctree(XTrain,YTrain);
%ROC CURVE 
figure
[labels,scores]=resubPredict(knn_model);                    %sınıflandırma puanlarını ve etiketlerini döndürür
[s1,m1,T,AUCKnn] = perfcurve(knn_model.Y,scores(:,2),'1');           %   1 pozitif değeri o yüzden seçildi
plot(s1,m1,'LineWidth',3)
hold on
[labels,scores]=resubPredict(nbModel);                    %sınıflandırma puanlarını ve etiketlerini döndürür
[s1,m1,T,AUCnb] = perfcurve(nbModel.Y,scores(:,2),'1');           %   1 pozitif değeri o yüzden seçildi
plot(s1,m1,'LineWidth',3)
hold on
[labels,scores]=resubPredict(svmModel);                    %sınıflandırma puanlarını ve etiketlerini döndürür
[s1,m1,T,AUCsvm] = perfcurve(svmModel.Y,scores(:,2),'1');           %   1 pozitif değeri o yüzden seçildi
plot(s1,m1,'LineWidth',3)
hold on
[labels,scores]=resubPredict(dtModel);                    %sınıflandırma puanlarını ve etiketlerini döndürür
[s1,m1,T,AUCdt] = perfcurve(dtModel.Y,scores(:,2),'1');           %   1 pozitif değeri o yüzden seçildi
plot(s1,m1,'LineWidth',3)
legend('KNN', 'Naive B', 'SVM', 'Desicion Tree');