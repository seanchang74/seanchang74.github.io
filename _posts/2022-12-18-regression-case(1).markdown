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

### 分析步驟
- [資料清理與視覺化圖表](#cleaning_visualization)
- [統計分析與特徵相關性分析](#advanced_analysis)
- [資料分割與建置迴歸模型](#model_design)
- [模型效能評估](#model_evaluation)
- [預測結果分析](#result)
    1. [線性迴歸](#linear_regression)
    2. [多項式線性迴歸](#poly_regression)
    3. [L2 正規化](#ridge_regularization)
    4. [Random Forest](#random_forest)
    5. [PCA降維後再訓練](#pca_compression)

<h2 id="cleaning_visualization">資料清理與視覺化圖表</h2>
在拿到一個全新的資料集後，我會先做以下幾個步驟處理:
1. 利用`df.isnull().sum()`檢查每個欄位的狀況，看是否有遺漏值。本資料集並沒有任何缺失值，因此不討論遺漏值填補的問題。
2. 接著使用`df.info()`查看各欄位的資料型態與欄位名稱，移除一些無意義的欄位(ex. **id**)。相似程度過高的欄位則可以進行欄位合併(ex. **citympg**和**highwaympg**)
3. 使用Seaborn與Matplotlib進行簡易的視覺化呈現

![Visualize_Feature_Price](/img/in-post/regression_case(1)/feature_price__corr.png)
<small class="img-hint">使用迴歸圖顯示各特徵與**price**之間的相關性</small> 

各特徵與price欄位之間的相關性
- 正相關: wheelbase, carlength, carwidth, curbweight, enginesize,
boreratio, horsepower
- 無明顯相關性: carheight, stroke, compressionratio, peakrpm
- 負相關: avg_mpg

![Visualize_Feature_Category](/img/in-post/regression_case(1)/category_distri.png)
<small class="img-hint>使用計數圖呈現object型態特徵中unique值的占比情形</small>
[Price_Distribution](/img/in-post/regression_case(1)/price_distri.png)
<small class="img-hint>使用長條圖的方式，展現price於各區間的分布情形</small>

根據上圖可以得知price主要集中於低價位，以**5000~15000**區間居多
4. 針對object型態的資料，採用編碼方式進行轉換。在進行編碼轉換時，我個人比較建議使用**One-hot encoding**，因為**label encoding**
是將值對應一個特定數字，而數字本身的大小並無任何意義，但在模型訓練時可能會產生判斷特徵之間的優劣迷思。 
經過 One-hot encoding 之後，特徵數從 24 種上升到 67 種
---
<h2 id="advanced_analysis">統計分析與特徵相關性分析</h2>
<h2 id="model_design">資料分割與建置迴歸模型</h2>
<h2 id="model_evaluation">模型效能評估</h2>
<h2 id="result">預測結果分析</h2>

---

## 參考資料

1. [TQC+ Python 3.x機器學習基礎與應用特訓教材](https://www.books.com.tw/products/0010888910)
2. [Regression-Visualization-Ensemble-93.65%](https://www.kaggle.com/code/aniket1993/regression-visualization-ensemble-93-65)