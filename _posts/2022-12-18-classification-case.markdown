---
layout:     post
title:      "機器學習分析個案(2)"
subtitle:   "判斷是否為心臟病個案(分類判斷)"
date:       2022-12-18 19:00:00
author:     "SeanChang"
header-img: "img/post-bg-classification-case(1).jpg"
tags:
    - MachineLearning
    - Classification
---
> 資料來源: [網址](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

## 分析步驟
1. [資料清洗](#cleaning)
2. [資料視覺化](#visualization)
3. [統計分析與特徵相關性分析](#advanced_analysis)
4. [資料分割與建置分類模型](#model_design)
5. [模型訓練](#model_training)
6. [模型評估](#model_evaluation)
7. [結論](#conclusion)

<h2 id="cleaning">資料清洗</h2>
1. 檢查是否存在遺漏值，如果有遺漏值我傾向使用中位數填補 (本資料集無遺漏值)
2. 檢查各欄位中數值分布情形，如果 nunique 值過小，代表欄位中重複值的比例偏高 (本資料集中除了 BMI 欄位外，大部分資料欄位中，皆有大量重複值的情形)
3. 檢查每一欄位的dtype，搭配欄位說明檔案進行資料轉換

處理邏輯: 
1. map 方法將布林型態欄位中的 Yes 值轉換成 1，No 值轉換成 0，所謂布林型態欄位就是數值僅有 Yes 與 No 兩種。 
2. 針對有大小順序的欄位(例如: GenHealth, AgeCategory)進行 Label Encoding。
3. 其餘的 object 型態資料則使用One-Hot Encoding 解決，經過一系列的處理後，特徵數目從原本的18種增加至27種。

---

<h2 id="visualization">資料視覺化</h2>
- 根據目標特徵 (本資料集為HeartDisease欄位)分布情形繪製一張圓餅圖
![Target_Distri](/img/in-post/classification_case(1)/target_distri.png)
<small class="img-hint">圖1: 有無心臟病的人數分布圓餅圖</small>

由上圖可以發現，本資料集大部分皆無心臟病其比例高達**91.4%**，可見資料分布極不平均。因此，較不建議僅使用 accuracy 去評判模型優劣。

- 此外，針對有無心臟病的人分別取樣 20000 筆資料，查看他們其餘特徵的分布情形，希望發現特徵與患有心臟病之間是否存在相關性。 

    >由於BMI特徵不重複值較多，故採用**stripplot**呈現，其餘皆使用**countplot**呈現計數狀況即可。
  
    - BMI
    ![BMI_Plot](/img/in-post/classification_case(1)/stripplot.png)
    <small class="img-hint">圖2: BMI 與 HeartDisease 之間關係圖</small>
    - Smoking
    ![Smoking_Plot](/img/in-post/classification_case(1)/smoking.png)
    <small class="img-hint">圖3: Smoking 與 HeartDisease 之間關係圖</small>
    - AlcoholDrinking
    ![AlcoholDrinking_Plot](/img/in-post/classification_case(1)/alcohol.png)
    <small class="img-hint">圖4: AlcoholDrinking 與 HeartDisease 之間關係圖</small>
    - Stroke
    ![Stroke_Plot](/img/in-post/classification_case(1)/stroke.png)
    <small class="img-hint">圖5: Stroke 與 HeartDisease 之間關係圖</small>
    - PhysicalHealth
    ![PhysicalHealth_Plot](/img/in-post/classification_case(1)/physical_health.png)
    <small class="img-hint">圖6: PhysicalHealth 與 HeartDisease 之間關係圖</small>
    - MentalHealth
    ![MentalHealth_Plot](/img/in-post/classification_case(1)/mental_health.png)
    <small class="img-hint">圖7: MentalHealth 與 HeartDisease 之間關係圖</small>
    - DiffWalking
    ![DiffWalking_Plot](/img/in-post/classification_case(1)/diffwalking.png)
    <small class="img-hint">圖8: DiffWalking 與 HeartDisease 之間關係圖</small>
    - Sex
    ![Sex_Plot](/img/in-post/classification_case(1)/sex.png)
    <small class="img-hint">圖9: Sex 與 HeartDisease 之間關係圖</small>
    - AgeCategory
    ![AgeCategory_Plot](/img/in-post/classification_case(1)/agecategory.png)
    <small class="img-hint">圖10: AgeCategory 與 HeartDisease 之間關係圖</small>
    - Race
    ![Race_Plot](/img/in-post/classification_case(1)/race.png)
    <small class="img-hint">圖11: Race 與 HeartDisease 之間關係圖</small>
    - Diabetic
    ![Diabetic_Plot](/img/in-post/classification_case(1)/diabetic.png)
    <small class="img-hint">圖12: Diabetic 與 HeartDisease 之間關係圖</small>
    - PhysicalActivity
    ![PhysicalActivity_Plot](/img/in-post/classification_case(1)/physical_act.png)
    <small class="img-hint">圖13: PhysicalActivity 與 HeartDisease 之間關係圖</small>
    - GenHealth
    ![GenHealth_Plot](/img/in-post/classification_case(1)/genhealth.png)
    <small class="img-hint">圖14: GenHealth 與 HeartDisease 之間關係圖</small>
    - SleepTime
    ![SleepTime_Plot](/img/in-post/classification_case(1)/sleeptime.png)
    <small class="img-hint">圖15: SleepTime 與 HeartDisease 之間關係圖</small>
    - Asthma
    ![Asthma_Plot](/img/in-post/classification_case(1)/asthma.png)
    <small class="img-hint">圖16: Asthma 與 HeartDisease 之間關係圖</small>
    - KidneyDisease
    ![KidneyDisease_Plot](/img/in-post/classification_case(1)/kidney.png)
    <small class="img-hint">圖17: KidneyDisease 與 HeartDisease 之間關係圖</small>
    - SkinCancer
    ![SkinCancer_Plot](/img/in-post/classification_case(1)/skin.png)
    <small class="img-hint">圖18: SkinCancer 與 HeartDisease 之間關係圖</small>

---

<h2 id="advanced_analysis">統計分析與特徵相關性分析</h2>
<h3>特徵相關性分析</h3>
根據上述圖2 ~ 圖18，可以看到以下相關性:
- 正相關: Smoking, Stroke, PhysicalHealth, DiffWalking, AgeCategory, Diabetes, Asthma, KidneyDisease, SkinCancer
    - 在 SleepTime 欄位中，睡太久或睡太少都有較高機率得心臟病，其中 6 ~ 8 小時的睡眠較佳。
    - 在 Sex 欄位中可以看到，男性得心臟病的機率大於女性。
    - 在 Race 欄位中可以看到，白人與印地安人得心臟病的機率大於其他種族。 
- 負相關: AlcoholDrinking, PhysicalActivity, GenHealth
- 無明顯相關: BMI, MentalHealth

個人觀點:
1. 大部分的特徵相關性其實透過常識判斷都算可以理解，那些經常運動、身體狀況越好的人，越不容易得心臟病;相對，如果有抽菸、中風、糖尿病等等狀況，越容易得心臟病。但最令我感到驚訝的是，喝酒並沒有增加患有心臟病的機率!如果不是親眼看見還真難令人相信。
2. 另外，資料顯示性別與種族也是影響是否容易得心臟病的關鍵指標。但我個人認為資料會如此呈現，主要是與個人的飲食習慣有關。較高比例的歐美白人，攝取速食或是較重口味的食物頻率比其他種族高，所以導致發生這種現象。同樣，男性較女性也是如此，所以我認為性別或種族本身並不是提高得心臟病的因素，而是個人的飲食習慣。
3. 我有嘗試刪除無相關特徵來提高模型效果，但效果並沒有太大改善，因此選擇以全部 27 種特徵作為後續模型的訓練樣本。

<h3>敘述性統計分析</h3>
1. 透過 describe 函數，檢查 float 型態資料的最大最小值、四分位數、標準差等資訊。
2. 接著將數值型欄位進行標準化動作，避免各欄位的尺度大小不一，影響離群值的判斷。
3. 將欄位繪製箱型圖檢視

![Statistic_Plot](/img/in-post/classification_case(1)/statistic.jpg)
<small class="img-hint">圖 19: 數值型欄位的箱型圖資訊</small>

>根據圖 19 可發現資料大都集中於前 1/3，如果結合前面畫的 countplot會有更明顯的感受，但這些離群值應該拿掉嗎?
> 
>我覺得不妥，因為大部分這些「離群值」都是患有心臟病的資料，如果拿掉當然訓練效果會變好，但這個模型就失去訓練的意義了!

---

<h2 id="model_design">資料分割與建置分類模型</h2>
由於是分類資料集，為了有更好的訓練效果，在切分資料集的時候添加`stratify`參數，可於切分時會根據訓練與測試集的大小比例，彈性分配適當數目的類別資料。我是採取8:2策略，其中訓練集占全部資料集的80% 共255836 筆，而測試集占剩下的 20%，共 63959 筆。

模型訓練將會分成以下幾種方法:
1. 羅吉斯迴歸 (分別採用 OvR 及 OvO 策略)
2. 支持向量機 (使用linear, rbf, poly, sigmoid 四種不同 kernel 函數測試)
3. Naïve Bayes
4. 隨機森林
5. KNN

---

<h2 id="model_training">模型訓練</h2>
- 羅吉斯迴歸
    1. 先使用迴圈的方式擬和不同類型的 C，繪製 score 趨勢表
    ![Logistic_Parameter](/img/in-post/classification_case(1)/logistic_c.png)
    <small class="img-hint">圖 20: 不同 C 值下羅吉斯迴歸的分類效果</small>
    2. 圖 20 發現，C = 0.1 時有不錯的擬和效果，套用 OvR 及 OvO 策略，score 相差不大皆為 0.915 左右。
- 支持向量機
    1. 原本使用 `GridSearchCV` 嘗試不同的組合，但組合約有 100 種，尋找及擬和時間過於漫長。
    2. 改用 `RandomizedSearchCV` 方法，僅從 100 個組合中隨機挑 30 種來擬和，同時開啟 `verbose` 功能(**如果想要正常顯示 `n_jobs` 只能設成1**)，這樣可以看清楚執行過程，但所需時間依舊過於漫長。 
    > - 非線性的核函數擬和一次都是數小時以上
    > - 設定 `n_jobs = -1` 時，程式會動用系統中所有的 CPU 去擬和模型，有平行處理的效果。同時 **CPU 使用率會直接到100%左右，記憶體也是**。雖然這樣會加速程式執行，但 `verbose` 輸出會消失，也就是說你無法知道程式執行的進度。
    3. 捨棄不同的 kernel 函數，僅使用 linear svc 調整 C 值進行擬和
    ![Linear_SVM_Parameter](/img/in-post/classification_case(1)/linear_svm_c.png)
    <small class="img-hint">圖 21: 不同 C 值下線性SVM的分類效果</small>

    > 挑選 C = 0.1 進行模型擬和，score 為 0.91 左右。
- Naïve Bayes
  
    執行時間非常快速，大概是按下執行鍵模型就完成了，我是使用 `GaussianNB`雖然有 `var_smoothing` 可供調整，但我不太清楚該如何調整會有較好的效果，於是直接使用預設值 score 為 0.813。

- 隨機森林
    1. 由於隨機森林是建構在決策樹的基礎上，於是先使用決策樹分類器，找出樹的深度應該設多少層才會有最好的分類效果。
    ![Random_Forest_Parameter](/img/in-post/classification_case(1)/random_forest_depth.png)
    <small class="img-hint">圖 22: 調整樹的深度，檢視決策樹的分類效果</small>
    >由圖 22 發現，樹高度設定為 6 層有不錯的效果，雖然第 5 層的 validation score 較高，但 test score 偏低，不太適合。
    2. 使用 **Graphviz** 套件(需額外自行安裝)視覺化決策樹的內部結構，由於圖片十分龐大，僅擷取一部分顯示。
    ![Decision_Tree](/img/in-post/classification_case(1)/tree.png)
    <small class="img-hint">圖 23: 視覺化決策樹分類過程</small>
    3. 接著，修改樹的數目，讓隨機森林分類器有更好的分類效果
    ![Tree_Num](/img/in-post/classification_case(1)/tree_num.png)
    <small class="img-hint">圖 24: 調整樹的數目，檢視隨機森林的分類效果</small>
    > 由圖 24 發現，validation 與 test 成績並沒有交集，於是只好挑 100 來擬和模型，因為兩個分數相差最近，較不易產生 overfit 情況。
- KNN
    1. 調整 k 的數目，先從 1~8 開始調整，檢查 validation 與 test 成績
    ![KNN_1_8](/img/in-post/classification_case(1)/knn_1_8.png)
    <small class="img-hint">圖 25: k = 1~8 時，檢視 KNN 的分類效果</small>
    2. 由圖 25 發現，accuracy 仍有小幅提升空間，於是我決定繼續調整 k 的數目，改成 k = 9 ~ 15。
    ![KNN_9_15](/img/in-post/classification_case(1)/knn_9_15.png)
    <small class="img-hint">圖 26: k = 9~15 時，檢視 KNN 的分類效果</small>
    > 由圖 26 發現，繼續提升 k 的數目依舊會提升模型的 accuracy，但同時也會大幅增加模型的訓練時間，相比之下繼續提高 k 的數目顯然不太實際。在經過一番考量之後，決定使用 **k = 12** 來擬和模型。

--- 

<h2 id="model_evaluation">模型評估</h2>
主要透過`混淆矩陣`及`Classification Report`兩種方式，評估分類模型的效能及實際可用性。
- 羅吉斯迴歸 (C = 0.1)
    - 混淆矩陣
    ![Logistic_Matrix](/img/in-post/classification_case(1)/logistic_confuse.jpg)
    <small class="img-hint">圖 27: 羅吉斯迴歸的混淆矩陣</small>
    - Classification Report
    
    |    | **precision** | **recall** | **f1-score** | **support** |
    |:------------:|:-------------:|:----------:|:------------:|:------------:|
    | 0            | 0.92          | 0.99       | 0.96         | 58484        |
    | 1            | 0.53          | 0.12       | 0.19         | 5475         |
    | accuracy     |               |            | 0.92         | 63959        |
    | macro avg    | 0.72          | 0.55       | 0.57         | 63959        |
    | weighted avg | 0.89          | 0.92       | 0.89         | 63959        |
    
    - 個人觀點
      1. 乍看之下，accuracy 高達 0.92，貌似表現優異，但如果仔細看一下就會發現大部分都是預測結果為 0 的貢獻，平均下來整體的數據才會如此亮眼。
      2. 可值得慶祝的是，如果模型通知你可能會得心臟病，有高達一半的機率是對的，但同時，由於 recall 值偏低，如果此模型真正用於醫療上，可能會有許多患者無法在心臟病早期時受到治療，進而導致病情惡化。
- 支持向量機 (線性kernel)
    - 混淆矩陣
      ![Linear_SVM_Matrix](/img/in-post/classification_case(1)/linear_svm_confuse.png)
      <small class="img-hint">圖 28: 線性SVM的混淆矩陣</small>
    - Classification Report

  |    | **precision** | **recall** | **f1-score** | **support** |
      |:------------:|:-------------:|:----------:|:------------:|:------------:|
  | 0            | 0.92          | 1       | 0.96         | 58484        |
  | 1            | 0.61          | **0.03**       | 0.07         | 5475         |
  | accuracy     |               |            | 0.92         | 63959        |
  | macro avg    | 0.76          | 0.52       | 0.51         | 63959        |
  | weighted avg | 0.89          | 0.92       | 0.88         | 63959        |
    
    - 個人觀點
      1. 一樣有透過預測為 0 而美化整體數據的現象。 
      2. Recall 值比上面的羅吉斯迴歸更低僅有約 0.03，也就是說真正有心臟病的患者幾乎檢測不出來。雖然模型有 0.92 的 accuracy，但從實際層面上，根本無法使用。因為我們是要一個在少數有異常狀況時，能成功檢測出來的模型，而不是不管有沒有狀況，都是顯示無異常。
- Gaussian Naïve Bayes
  - 混淆矩陣
    ![Naive_Bayes_Matrix](/img/in-post/classification_case(1)/naive_bayes_confuse.png)
    <small class="img-hint">圖 29: Naïve Bayes 的混淆矩陣</small>
  - Classification Report

  |    | **precision** | **recall** | **f1-score** | **support** |
        |:------------:|:-------------:|:----------:|:------------:|:------------:|
  | 0            | 0.95          | 0.84       | 0.89         | 58484        |
  | 1            | 0.24          | **0.57**       | 0.34         | 5475         |
  | accuracy     |               |            | 0.81         | 63959        |
  | macro avg    | 0.6          | 0.7       | 0.62         | 63959        |
  | weighted avg | 0.89          | 0.81       | 0.84         | 63959        |
  
  - 個人觀點
    1. 在 accuracy 上雖然比前兩個模型低了不少，但 recall 的部分卻有大幅度的提升，達到了 0.57 是個不錯的成績。
    2. 比較可惜的是精準度並不高，可能會有不少沒有心臟病的病人，被誤判成有心臟病。
- 隨機森林 (`n_estimators` = 100、`max_depth` = 6)
  - 混淆矩陣
    ![Random_Forest_Matrix](/img/in-post/classification_case(1)/random_forest_confuse.png)
    <small class="img-hint">圖 30: 隨機森林的混淆矩陣</small>
  - Classification Report

  |    | **precision** | **recall** | **f1-score** | **support** |
          |:------------:|:-------------:|:----------:|:------------:|:------------:|
  | 0            | 0.92          | 1       | 0.96         | 58484        |
  | 1            | 0.71          | 0.02       | 0.03         | 5475         |
  | accuracy     |               |            | 0.92         | 63959        |
  | macro avg    | 0.81          | 0.51       | 0.49         | 63959        |
  | weighted avg | 0.9          | 0.92       | 0.88         | 63959        |
  
  - 個人觀點 
  1. 效果與支持向量機差不多，只不過 1 的 precision 更高了一些，但實際層面上距離能使用還有一大段距離。
- KNN (`n_neighbors` = 12)
  - 混淆矩陣
    ![KNN_Matrix](/img/in-post/classification_case(1)/knn_confuse.png)
    <small class="img-hint">圖 31: KNN的混淆矩陣</small>
  - Classification Report

  |    | **precision** | **recall** | **f1-score** | **support** |
            |:------------:|:-------------:|:----------:|:------------:|:------------:|
  | 0            | 0.92          | 1       | 0.95         | 58484        |
  | 1            | 0.45          | 0.03       | 0.05         | 5475         |
  | accuracy     |               |            | 0.91         | 63959        |
  | macro avg    | 0.69          | 0.51       | 0.5         | 63959        |
  | weighted avg | 0.88          | 0.91       | 0.88         | 63959        |
  
  - 個人觀點
    1. 在 1 的 precision 上比支持向量機與隨機森林還要低，此外recall 表現與不怎麼樣，看來此資料集不太適合使用 KNN。

---

<h2 id="conclusion">結論</h2>
1. 本資料集的數目偏多(達約 32 萬筆)，導致一些時間複雜度偏高的模型訓練時間無法預期。
2. 本資料集會有許多離群值的情況，從前面的敘述統計階段就可以發現，但在分類資料集中，往往這些離群值才是整個資料集的核心，所以敘述統計在此資料集其實影響有限。
3. 本次作業的目的是根據一些個人身體狀況、年齡、睡眠時間等特徵數據，判斷此人有無心臟病。但可惜的是，兩種資料分布比例差距過大，在判斷什麼人沒有心臟病這點上，大部分的模型都表現不錯，但成功找出什麼人會有心臟病的表現上就不太理想。
4. 除了使用全部資料集下去訓練之外，在附上的程式檔中，我有額外做一個實驗，改成有無心臟病各取 20000 筆資料，合併成一個新的資料集做訓練，accuracy 雖然降至 0.77 左右，但在預測 0 與 1 上都有維持 0.75以上的水準，算是有滿足實際層面上的需求。

---

## 參考資料

1. [TQC+ Python 3.x機器學習基礎與應用特訓教材](https://www.books.com.tw/products/0010888910)
2. [HeartDisease: Visualisation 📊 Classification 🔮](https://www.kaggle.com/code/ricksan4ez/heartdisease-visualisation-classification)