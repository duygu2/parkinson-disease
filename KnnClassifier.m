folder = 'C:\Users\Duygu\Desktop\Veri Madenciliği\parkinson';
fullFileName = fullfile(folder, 'parkinsons.csv');

parkinsonData = readtable(fullFileName);
data= parkinsonData(:,[2:24]);

%30 u test olarak kullanıldı.
pt = cvpartition(data.status,"HoldOut",0.3);

%eğitim ve test verileri veriden ayrıştırıldı.
pdTrain = data(training(pt),:);
pdTest = data(test(pt),:);

%KNN model fonk uygulandı. Status tablodaki 0 ve 1'ler. Sağlıklı veya Hasta
knn_model = fitcknn(pdTrain,'status','NumNeighbors',5);

%verilere tahmin edilen yanıtlar oluşturur.
tahmin = predict(knn_model,pdTest);

%Model oluşturmak için oluşan kayıp 

trainError = resubLoss(knn_model)

%cross validation hesaplaması: modelin pratikte ne kadar doğrulukta
%çalışacağını kestirir

cvMdl = crossval(knn_model)

%cross validation hesaplaması sonucu CVMdl tarafından elde edilen ortalama sınıflandırma hatası.

cvtrainError = kfoldLoss(cvMdl)

%cross validation eğitim doğruluğu

cvtrainAccuracy = 1-cvtrainError

%model ve test sonuçları karşılaştırılarak elde edilen hata

newError = loss(knn_model,pdTest,'status')

%doğruluk hesaplaması

newAccuracy = 1-newError


%gerçek etiketlerden ve tahmin edilen etiketlerden bir karışıklık matrisi grafiği oluşturur 
%Karışıklık matrisinin satırları gerçek sınıfa karşılık gelir ve sütunlar tahmin edilen sınıfa karşılık gelir.
%Köşegen ve köşegen dışı hücreler, sırasıyla doğru ve yanlış sınıflandırılmış gözlemlere karşılık gelir.
figure
confusionchart(tahmin, pdTest.status);

confmat = confusionmat(pdTest.status,tahmin);
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



%ROC CURVE 
figure
[labels,scores]=resubPredict(knn_model);                    %sınıflandırma puanlarını ve etiketlerini döndürür
[s1,m1,T,AUC] = perfcurve(knn_model.Y,scores(:,2),'1');           %   1 pozitif değeri o yüzden seçildi
plot(s1,m1,'LineWidth',3)

AUC


