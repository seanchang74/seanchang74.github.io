---
layout:     post
title:      "機器學習分析個案(1)"
subtitle:   "預測汽車售價(迴歸分析)"
date:       2022-12-18 12:00:00
author:     "SeanChang"
header-img: "img/post-bg-regression-case(1).jpg"
tags:
    - MachineLearning
    - Regression
---
> 資料來源: [網址](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)

## 分析步驟
1. [資料清理與視覺化圖表](#cleaning_visualization)
2. [統計分析與特徵相關性分析](#advanced_analysis)
3. [資料分割與建置迴歸模型](#model_design)
4. [模型效能評估](#model_evaluation)
5. [預測結果分析](#result)

<h2 id="cleaning_visualization">資料清理與視覺化圖表</h2>
在拿到一個全新的資料集後，我會先做以下幾個步驟處理:
1. 利用`df.isnull().sum()`檢查每個欄位的狀況，看是否有遺漏值。本資料集並沒有任何缺失值，因此不討論遺漏值填補的問題。
2. 接著使用`df.info()`查看各欄位的資料型態與欄位名稱，移除一些無意義的欄位(ex. **id**)。相似程度過高的欄位則可以進行欄位合併(ex. **citympg**和**highwaympg**)
3. 使用Seaborn與Matplotlib進行簡易的視覺化呈現

![Visualize_Feature_Price](/img/in-post/regression_case(1)/feature_price_corr.png)
<small class="img-hint">使用迴歸圖顯示各特徵與price之間的相關性</small> 

各特徵與price欄位之間的相關性
- 正相關: wheelbase, carlength, carwidth, curbweight, enginesize,
boreratio, horsepower
- 無明顯相關性: carheight, stroke, compressionratio, peakrpm
- 負相關: avg_mpg

![Visualize_Feature_Category](/img/in-post/regression_case(1)/category_distri.png)
<small class="img-hint">使用計數圖呈現object型態特徵中unique值的占比情形</small>
![Price_Distribution](/img/in-post/regression_case(1)/price_distri.png)
<small class="img-hint">使用長條圖的方式，展現price於各區間的分布情形</small>

根據上圖可以得知price主要集中於低價位，以**5000~15000**區間居多
> 針對object型態的資料採用編碼方式進行轉換。在進行編碼轉換時，我個人比較建議使用**One-hot encoding**，因為label encoding是將值對應一個特定數字，而**數字本身的大小並無任何意義**，但在模型訓練時可能會產生判斷特徵之間的優劣迷思。 

經過 One-hot encoding 之後，特徵數從 24 種上升到 67 種
   
---

<h2 id="advanced_analysis">統計分析與特徵相關性分析</h2>
<h3>敘述性統計分析</h3>
1. 先使用`df.describe()`顯示數值型欄位的標準差, 四分位數, 最大最小值等基本資訊
2. 接著將數值型欄位進行標準化動作，避免各欄位的尺度大小不一，影響離群值的判斷
3. 將欄位繪製箱型圖檢視

![Statistic](/img/in-post/regression_case(1)/statistic.jpg)
<small class="img-hint">數值型欄位的箱型圖資訊</small>
個人觀點:
1. 由上圖可以看出，在compressionratio這個特徵中，**離群值明顯大於整體資料**，這很容易造成模型誤判，降低精確程度
2. 對於 curbweight, peakrpm 這些特徵如果沒有經過標準化，很容易因為特徵本身尺度偏大，造成標準差數值偏高，讓人誤以為資料本身離散程度高。

<h3>特徵相關性分析</h3>
1. 生成相關係數矩陣，使用預設的皮爾森相關係數
2. 剔除對 price 欄位相關性過低的特徵(絕對值<0.5)
3. 剔除兩個欄位彼此相似性過高的特徵(絕對值>0.9)

> 在經過篩選之後，僅取出12種特徵用來做後面的模型訓練

繪製熱點圖檢視

![Corr_Matrix](/img/in-post/regression_case(1)/corr_matrix.png)
<small class="img-hint">篩選出來的特徵彼此之間的相關性比較</small>

---

<h2 id="model_design">資料分割與建置迴歸模型</h2>
資料集分割使用 Scikit-learn 內建的 `train_test_split` 切分資料集，其中訓練集占全部資料集 80%，共 164 筆；測試集為剩下的 20%，共41 筆。模型訓練除了第一種方式是採用全部 67 種特徵，其餘方法僅採用篩選出來的 12 種特徵。

模型訓練將會分成以下幾種方法:
- 以全部 67 種特徵訓練線性迴歸模型
- 僅以篩選出來的 12 種特徵訓練線性迴歸模型
- 多項式迴歸(degree 僅測試 1~5)
- 經過 L2 正規化調整後的線性迴歸模型
- 隨機森林迴歸法
- 透過 PCA 將特徵壓縮成一維陣列再跑線性迴歸

---

<h2 id="model_evaluation">模型效能評估</h2>
評估標準有 **R^2** 分數, **MAE**, **MSE** 及 **RMSE** 指標，其中訓練集與測試集的數值都會一併顯示。由於版面有限，將以編號呈現各模型訓練方法。

|         | **R^2分數** | **R^2分數** | **MAE** | **MAE** | **MSE**   | **MSE**     | **RMSE** | **RMSE** |
|:----------------:|:---------:|:---------:|:-------:|:-------:|:---------:|:-----------:|:--------:|:---------:|
|                  | 訓練        | 測試        | 訓練      | 測試      | 訓練        | 測試          | 訓練       | 測試        |
| 方法(壹)            | 0.973     | 0.899     | 959     | 1923    | 1629248   | 8002007     | 1276     | 2829      |
| 方法(貳)            | 0.83      | 0.84      | 2252    | 2473    | 10112470  | 12609007    | 3180     | 3551      |
| 方法(參) degree = 2 | 0.746     | -0.981    | 2906    | 7088    | 15147390  | 156364800   | 3892     | 12505     |
| 方法(參) degree = 3 | 0.998     | -9002     | 95      | 225453  | 106313    | 71069440000 | 326      | 843027    |
| 方法(肆)            | -3        | 0.84      | 13223   | 2473    | 234484544 | 12609007    | 15313    | 3551      |
| 方法(伍)            | 0.984     | 0.958     | 622     | 1253    | 971234    | 3339538     | 986      | 1827      |
| 方法(陸)            | 0.682     | 0.753     | 2772    | 2956    | 18944281  | 19489660    | 4353     | 4415      |

個人觀點:
1. 根據上述評估指標，可以看出方法(參)有嚴重的overfit現象，尤其是從 degree=3 開始，其 MAE, MSE, RMSE 的數值呈現巨幅攀升，因此**多項式迴歸方法不適合本資料集**。
2. 除了方法(參)有嚴重過擬和現象，方法(壹)也有輕微的過擬和，雖然所有指標在訓練與測試方面都有不錯的表現，但訓練與測試之間的差距依舊不小，如果採用其他模型效果可能更加嚴重，因此改用**篩選出來的 12 種特徵做後續訓練**。
3. 在方法(肆)中，不知道是我程式寫錯的關係還是正常現象，在經過正規化之後，訓練集的分數居然遠低於測試集，這個現象實屬異常。
4. 在方法(陸)中，我嘗試藉由 PCA 的方式將多個特徵進行壓縮，想說如果壓縮成一維陣列再找尋線性方程式來擬和，效果可能較好。但實際上並非如此，我覺得應該是**壓縮之後特徵的顯著性大幅降低**，因此對於預測效果大打折扣。
5. 根據上表，可以得知在本資料集中，由**隨機森林迴歸法的表現最佳**，且訓練與測試的差距較小，沒有發現過擬和現象。

---

<h2 id="result">預測結果分析</h2>
主要以視覺化的方式，呈現測試集中預測值與真實值。其中由於多項式迴歸方法效果不太理想，因此並**沒有額外畫圖**。
1. 以全部 67 種特徵訓練線性迴歸模型
![Method_1](/img/in-post/regression_case(1)/method_1.png)
<small class="img-hint">方法(壹)的真實值與預測值的比較圖</small>
2. 僅以篩選出來的 12 種特徵訓練線性迴歸模型
![Method_2](/img/in-post/regression_case(1)/method_2.png)
<small class="img-hint">方法(貳)的真實值與預測值的比較圖</small>
3. 經過 L2 正規化調整後的線性迴歸模型
![Method_4](/img/in-post/regression_case(1)/method_4.png)
<small class="img-hint">方法(肆)的真實值與預測值的比較圖</small>
4. 隨機森林迴歸法
![Method_5](/img/in-post/regression_case(1)/method_5.png)
<small class="img-hint">方法(伍)的真實值與預測值的比較圖</small>
5. 透過 PCA 將特徵壓縮成一維陣列再跑線性迴歸
![Method_6](/img/in-post/regression_case(1)/method_6.png)
<small class="img-hint">方法(陸)的真實值與預測值的比較圖</small>

個人觀點:
1. 有一個很特別的現象，在第 16 筆測試資料的時候，所有的線性迴歸模型都表現不太理想，預測值來到了 0 元左右，有點太誇張;另外，在第 25 筆測試資料的時候，也有很明顯的差距，線性迴歸模型的預測結果，都與真實值相差超過10000元以上，尤其是經過 PCA 壓縮後的模型，更是超過了 15000元，僅有隨機森林的預測結果相距是在 5000 元左右。
2. 本次模型訓練雖然使用不同的方法改善線性迴歸模型，但從預測結果趨勢圖來看，其實並沒有相差太多，整體價格走勢是差不多的，在幾個極端值的表現上，也沒有明顯差異。
3. 由結果來看，本資料集較適合決策樹類型的演算法(決策樹/隨機森林)，我個人猜測有可能是因為離群值過多，因此使用線性的方式較難找出資料之間的規律，雖然我這邊沒有列出決策樹的預測結果，但在筆記本中有提供程式參考。

在最大僅使用 4 層的決策樹迴歸就達到 R^2(training): 0.942, R^2(testing): 0.917 的優異成績，使用隨機森林後，更達到R^2(training): 0.984, R^2(testing): 0.958。

---

## 參考資料
1. [TQC+ Python 3.x機器學習基礎與應用特訓教材](https://www.books.com.tw/products/0010888910)
2. [Regression-Visualization-Ensemble-93.65%](https://www.kaggle.com/code/aniket1993/regression-visualization-ensemble-93-65)