#!/usr/bin/env python
# coding: utf-8

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

#conf = SparkConf().setAppName('Insight_Ingest_NLP_LogistcRgreesion1')
#.setMaster('yarn')
#sc = pyspark.SparkContext(conf=conf).getOrCreate()
#spark = SparkSession(sc)
spark = SparkSession.builder.appName('small_nlp_logistic').getOrCreate()

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df_review = spark.read.json("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/yelp_academic_dataset_review.json")
#df_review = df_review.limit(500)

import pyspark.sql.functions as f
df_review = df_review.filter("cool >=3 or useful >=3 or funny >=3") 
df_review = df_review.select("stars","text")
df_review = df_review.repartition(100)

from pyspark.sql import functions as F
df_review = df_review.withColumn("target", F.when( df_review.stars <=2,1 ).otherwise(0))
df_review.cache()

(train_set, test_set) = df_review.randomSplit([0.7, 0.3], seed = 1002)

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="text", outputCol="words")
cv = CountVectorizer(vocabSize=2**16, inputCol="words", outputCol='cv')
idf = IDF(inputCol='cv', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
#lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, cv, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(test_set)

lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)
#predictions = predictions.select('target','label', 'rawPrediction', 'probability', 'prediction')

from pyspark.sql.functions import col
prediction_1 = predictions.select('prediction','label','target',col("probability").cast("string"))
prediction_1.repartition(1).write.mode('overwrite').format('com.databricks.spark.csv').save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/logisticReg_pre.csv")

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
from pyspark.sql.types import FloatType
eva_pre = evaluator.evaluate(predictions)
df_eva = spark.createDataFrame([eva_pre], FloatType())
df_eva.write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/auc.csv")

evaluator.getMetricName()
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_set.count())
df_acc = spark.createDataFrame([accuracy], FloatType())
df_acc.write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/acc.csv")

from pyspark.sql.types import StringType, FloatType
list_coef = [float(round(x,4)) for x in lrModel.coefficients]
list_word = pipelineFit.stages[1].vocabulary
df_word_list = spark.createDataFrame(list_word,schema = StringType())
df_word_list.write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/word_list.csv")
sc.parallelize(list_coef).coalesce(1).saveAsTextFile("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/coef_list.txt")

#coe = lrModel.coefficients
#list_1 = []
#for i in range(len(lrModel.coefficients)):
#    list_temp = (str(pipelineFit.stages[1].vocabulary[i]),float(round(coe[i],5)))
#    list_1.append(list_temp)
#from pyspark.sql.types import FloatType, StringType
#df_coef = sqlContext.createDataFrame(list_1,["word","coef"]).filter("coef>=0.1").orderBy("coef", ascending = False)
#df_coef.write.mode('overwrite').format('com.databricks.spark.csv').save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/logisticReg_coef.csv")

lrModel.save("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/model/lrmodel_neo")
print(accuracy)