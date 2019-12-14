import sys
# 設定PYTHONPATH
sys.path.append('/usr/local/spark/python')
sys.path.append('/usr/local/spark/python/lib/py4j-0.10.7-src.zip')

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np

sf = SparkConf().setAppName("SparkSessionZipsExample")
sc = SparkContext(conf=sf)
print(sc.master)
# 印目前的路徑
# print(!echo $PWD)

print("開始匯入資料...")
# 匯入character-deaths.csv
rawDataWithHeader = sc.textFile("./pyspark project/character-deaths.csv")

# 移除第一筆欄位名稱
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x:x !=header)
# 移除雙引號 (")
rData = rawData.map(lambda x: x.replace("\"", ""))
# 取得每一列資料欄位
lines = rData.map(lambda x: x.split(","))
# 顯示資料筆數
print("共計：" + str(lines.count()) + "筆")


# 擷取feature特徵欄位
def extract_features(field, categoriesMap, featureEnd):
    # 擷取分類特徵欄位
    categoryIdx = categoriesMap[field[1]]  # 家族分類轉換為數值
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1  # 設定list相對應的位置是1

    # 擷取數值欄位
    numericalFeatures = [convert_float(field) for field in field[5: featureEnd]]

    # 回傳「分類特徵欄位」+「數字特徵欄位」
    return np.concatenate((categoryFeatures, numericalFeatures))


# 填補空值以0代替
def convert_float(x):
    return 0 if x == "" else x


# 擷取label標籤欄位
def extract_label(field):
    label = (convert_bi(field[2]))
    return label


# 填補空值以0代替，有值以1代替
def convert_bi(x):
    return 0 if x == "" else 1

# 建立categoriesMap家族分類字典，1個分類對應1個數字
categoriesMap = lines.map(lambda fields: fields[1]).distinct().zipWithIndex().collectAsMap()

# 建立訓練評估所需LabeledPoint資料
labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, categoriesMap, len(r))))

# 查看資料前處理結果
print(labelpointRDD.first())

# 以randomSplit隨機方式，依照3：1 (75％：25％) 比例，將資料分為train set與test set
(trainData, testData) = labelpointRDD.randomSplit([3, 1])

# 為加快程式的執行效率，將train set與test set暫存在記憶體中
trainData.persist()
testData.persist()

# 使用Spark MLlib支援的決策樹
model=DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={},impurity="entropy", maxDepth=5, maxBins=5)
# 使用model.predict對testDat作預測
score = model.predict(testData.map(lambda p: p.features))
# 將預測結果與真實label結合起來
scoreAndLabels = score.zip(testData.map(lambda p: p.label))
# 使用MulticlassMetrics做出confusionMatrix，計算Accuracy，Recall，Precision
metrics = MulticlassMetrics(scoreAndLabels)
print(metrics.confusionMatrix())
print("Accuracy = %s" % metrics.accuracy)
print("Recall = %s" % metrics.recall(0))
print("Precision = %s" % metrics.precision(0))

