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
<small class="img-hint">有無心臟病的人數分布圓餅圖</small>

由上圖可以發現，本資料集大部分皆無心臟病其比例高達91.4%，可見資料分布極不平均。因此，較不建議僅使用 accuracy 去評判模型優劣。

- 此外，針對有無心臟病的人分別取樣 20000 筆資料，查看他們其餘特徵的分布情形，希望發現特徵與患有心臟病之間是否存在相關性。 

>由於BMI特徵不重複值較多，故採用**stripplot**呈現，其餘皆使用countplot呈現計數狀況即可。

- BMI
![BMI_Plot](/img/in-post/classification_case(1)/stripplot.png)
<small class="img-hint">圖2: BMI 與 HeartDisease 之間關係圖</small>
> 補圖

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
<h2 id="model_evaluation">模型評估</h2>
<h2 id="conclusion">結論</h2>
## 參考資料

1. [Python資料科學學習手冊](https://www.books.com.tw/products/0010774364)
2. [初探機器學習：使用Python](https://www.books.com.tw/products/0010764445)
3. [新手村逃脫！初心者的 Python 機器學習攻略（iT邦幫忙鐵人賽系列書）](https://www.books.com.tw/products/0010867390)
4. [Coursera Machine Learning Class](https://www.coursera.org/learn/machine-learning)