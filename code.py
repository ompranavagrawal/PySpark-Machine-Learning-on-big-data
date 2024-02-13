#Author: Pranav Agrawal G01394901
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.functions import regexp_replace, trim, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline

from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('Assignment2').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

def countMissing(validData):
    missDataAll = []
    for c in validData.columns:
        condition = func.isnan(c) | func.col(c).isNull()
        count = func.count(func.when(condition, c))
        result = count.alias(c)
        missDataAll.append(result)
    missData = validData.select(missDataAll)
    return missData

def analyzeModel(model, testData, trainData):
    testPredictions = model.transform(testData)
    testAccuracy = MulticlassClassificationEvaluator(labelCol='fraudulent', metricName='accuracy').evaluate(testPredictions)
    trainPredictions = model.transform(trainData)
    trainAccuracy = MulticlassClassificationEvaluator(labelCol='fraudulent', metricName='accuracy').evaluate(trainPredictions)
    f1 = MulticlassClassificationEvaluator(labelCol='fraudulent', metricName='f1').evaluate(testPredictions)
    return testAccuracy, trainAccuracy, f1


data = spark.read.csv("hdfs:///user/pagrawa/fake_job_postings.csv", header=True, inferSchema=True)

validData = data.filter((data['fraudulent'] == 0) | (data['fraudulent'] == 1))
countMiss = countMissing(validData)
countMissDict = countMiss.collect()[0].asDict()

totalRows = validData.count()
missPercentages = {c: (v/totalRows) * 100 for c, v in countMissDict.items()}

with open("output/Columns/columns_with_missing_values.txt", 'w') as fp:
    for column, missCount in countMissDict.items():
        print(f"Column: {column}, Missing values: {missCount}, Percentage: {missPercentages[column]:.2f}%", file=fp)

dropColumns=[c for c, v in missPercentages.items() if v > 1]
cleanedValidData = validData.drop(*dropColumns)

patternForRemoval = '[^a-zA-Z0-9 ]'
patternForSpaces = ' +'
for col in cleanedValidData.columns:
    cleanedValidData = cleanedValidData.withColumn(col,trim(lower(regexp_replace(regexp_replace(col, patternForRemoval, ''),patternForSpaces,' '))))

majority = cleanedValidData.filter(cleanedValidData.fraudulent == 0)
minority = cleanedValidData.filter(cleanedValidData.fraudulent == 1)
fractionMaj = minority.count() / majority.count()
undersampledMaj = majority.sample(False, fractionMaj)
dataBalanced = undersampledMaj.union(minority)

tokenizer = Tokenizer(inputCol="description", outputCol="words")
stopwordsRemove= StopWordsRemover(inputCol="words", outputCol="filtered_words")

tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=21000)
idf = IDF(inputCol="raw_features", outputCol="features")
pipeline = Pipeline(stages=[tokenizer, stopwordsRemove, tf, idf])

tfidf = pipeline.fit(dataBalanced)
processedData = tfidf.transform(dataBalanced)

processedData = processedData.withColumn("fraudulent", func.col("fraudulent").cast("int"))
trainData, testData = processedData.randomSplit([0.7, 0.3], seed=42)

#Logistic Regression

lr = LogisticRegression(featuresCol='features', labelCol='fraudulent')

paramGridLR = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.0, 10, 15]) \
    .addGrid(lr.maxIter, [0, 3, 5]) \
    .addGrid(lr.elasticNetParam, [0.5]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol='fraudulent')
crossval = CrossValidator(estimator=lr, estimatorParamMaps=paramGridLR, evaluator=evaluator, numFolds=10, collectSubModels=True)

cVLRModel = crossval.fit(trainData)

with open("output/LR/output_LR.txt", 'w') as f:
    try:
        models = [model for fold in cVLRModel.subModels for model in fold]

    except TypeError:
        models = cVLRModel.subModels

    for i, model in enumerate(models):
        params = paramGridLR[i % len(paramGridLR)]
        params = paramGridLR[i % len(paramGridLR)]
        testAccuracyLR, trainAccuracyLR, f1LR=analyzeModel(model, testData, trainData)
        print(f"Params: {params} => Train accuracy: {trainAccuracyLR}, Test accuracy: {testAccuracyLR}, F1: {f1LR}", file=f)

    print("=====================================================", file=f)
    print("Best Params:", cVLRModel.bestModel.extractParamMap(), file=f)

#LinearSVC

svc = LinearSVC(featuresCol='features', labelCol='fraudulent')

paramGridSVC = ParamGridBuilder() \
    .addGrid(svc.maxIter, [3, 5]) \
    .addGrid(svc.aggregationDepth, [3]) \
    .addGrid(svc.regParam, [0, 10, 15]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol='fraudulent')
crossval = CrossValidator(estimator=svc, estimatorParamMaps=paramGridSVC, evaluator=evaluator, numFolds=10, collectSubModels=True)

cVSVCModel = crossval.fit(trainData)

with open("output/SVC/output_SVC.txt", 'w') as f:
    try:
        models = [model for fold in cVSVCModel.subModels for model in fold]

    except TypeError:
        models = cVSVCModel.subModels

    for i, model in enumerate(models):
        params = paramGridSVC[i % len(paramGridSVC)]
        testAccuracySVC, trainAccuracySVC, f1SVC=analyzeModel(model, testData, trainData)
        print(f"Params: {params} => Train accuracy: {trainAccuracySVC}, Test accuracy: {testAccuracySVC}, F1: {f1SVC}", file=f)

    print("=====================================================", file=f)
    print("Best Params:", cVSVCModel.bestModel.extractParamMap(), file=f)

#RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol='fraudulent')

paramGridRF = ParamGridBuilder() \
    .addGrid(rf.numTrees, [40, 50, 60, 70]) \
    .addGrid(rf.maxDepth, [25, 30]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol='fraudulent')
crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGridRF, evaluator=evaluator, numFolds=10, collectSubModels=True)

cVRFCModel = crossval.fit(trainData)

with open("output/RFC/output_RFC.txt", 'w') as f:
    try:
        models = [model for fold in cVRFCModel.subModels for model in fold]
    except TypeError:
        models = cVRFCModel.subModels

    for i, model in enumerate(models):
        params = paramGridRF[i % len(paramGridRF)]
        testAccuracyRFC, trainAccuracyRFC, f1RFC=analyzeModel(model, testData, trainData)
        print(f"Params: {params} => Train accuracy: {trainAccuracyRFC}, Test accuracy: {testAccuracyRFC}, F1: {f1RFC}", file=f)

    print("=====================================================", file=f)
    print("Best Params:", cVRFCModel.bestModel.extractParamMap(), file=f)

#MultilayerPerceptronClassifier

mlp = MultilayerPerceptronClassifier(featuresCol='features', labelCol='fraudulent', layers=[21000, 24, 2])

paramGridMLP = ParamGridBuilder() \
    .addGrid(mlp.maxIter, [10, 15, 20]) \
    .addGrid(mlp.stepSize, [0.001, 0.01, 0.1]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol='fraudulent')
crossval = CrossValidator(estimator=mlp, estimatorParamMaps=paramGridMLP, evaluator=evaluator, numFolds=10, collectSubModels=True)

cVMLPModel = crossval.fit(trainData)

with open("output/MLP/output_MLP.txt", 'w') as f:
    try:
        models = [model for fold in cVMLPModel.subModels for model in fold]
    except TypeError:
        models = cVMLPModel.subModels

    for i, model in enumerate(models):
        params = paramGridMLP[i % len(paramGridMLP)]
        testAccuracyMLP, trainAccuracyMLP, f1MLP=analyzeModel(model, testData, trainData)
        print(f"Params: {params} => Train accuracy: {trainAccuracyMLP}, Test accuracy: {testAccuracyMLP}, F1: {f1MLP}", file=f)

    print("=====================================================", file=f)
    print("Best Params:", cVMLPModel.bestModel.extractParamMap(), file=f)

spark.stop()