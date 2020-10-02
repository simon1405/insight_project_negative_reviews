#!/usr/bin/env python
# coding: utf-8



import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

#conf = SparkConf().setAppName('Insight_Ingest_NLP_LogistcRgreesion1')
#.setMaster('yarn')
#sc = pyspark.SparkContext(conf=conf).getOrCreate()
#spark = SparkSession(sc)
spark = SparkSession.builder.appName('insight_tabs').getOrCreate()

input_path = "s3://cf-templates-efn3brqcqavf-us-east-2/yelp/"
output_path = "s3://cf-templates-efn3brqcqavf-us-east-2/yelp/output/tables/"

df_review = spark.read.json(input_path +"yelp_academic_dataset_review.json")
df_business = spark.read.json(input_path +"yelp_academic_dataset_business.json")
df_user = spark.read.json(input_path +"yelp_academic_dataset_user.json")

df_bsn_list = df_business.groupBy(["city","state"]).count().orderBy("count",ascending = False)
df_bsn_list.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"business_list.csv")

df_review_neg = df_review.filter("stars<=2")

import pyspark.sql.functions as F

df_1_city = df_business.filter((F.col('city') == "Las Vegas" )&(F.col('state') == "NV"))
df_2_city = df_business.filter((F.col('city') == "Toronto" )&(F.col('state') == "ON"))
df_3_city = df_business.filter((F.col('city') == "Phoenix" )&(F.col('state') == "AZ"))
df_cities = df_1_city.union(df_2_city).union(df_3_city).drop("text","stars","hours", "is_open")
df_tab1 = df_cities.join(df_review_neg, on = "business_id", how = "right").select('city', 'latitude', 'longitude', 'name', 'state','stars')
df_tab1.columns
#df_tab1.show(10, False)
df_tab1.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"tab1.csv")
#
###tab2
df_re_bsn = df_business.join(df_review_neg, on="business_id", how="inner")
df_neg_bsn = df_re_bsn.groupBy("name").count().orderBy("count", ascending = False).withColumnRenamed("count","num_neg_re")
df_re_all_bus = df_review.join(df_business, on="business_id", how="inner")
df_all_bsn = df_re_all_bus.groupBy("name").count().orderBy("count", ascending = False).withColumnRenamed("count","num_all_re")
df_re_com = df_all_bsn.join(df_neg_bsn, on = "name", how = "inner")
df_tab2 = df_re_com.withColumn("percent", df_re_com.num_neg_re/df_re_com.num_all_re*100.0).orderBy("num_neg_re",ascending = False)

#df_tab2.show(10, False)
df_tab2.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"tab2.csv")
com_list = ["McDonald's","Starbucks","Chipotle Mexican Grill","Dunkin'","Buffalo Wild Wings","Denny's","Panera Bread","Pizza Hut","Taco Bell","Wendy's"]

df_10_bsn = df_re_bsn#filter(df_re_bsn.name.isin(com_list)).select("text")

from pyspark.sql.functions import split,col,explode,count

df_tab4=df_10_bsn.withColumn('words',split(col('text'),' '))\
.withColumn('word',explode(col('words')))\
.drop('text','words').groupBy('word').agg(count('word')\
 .alias('count')).orderBy('count',ascending=False)

df_tab4.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"tab4.csv")

for i in range(10):
    df_tab3_iter=df_10_bsn.filter(df_10_bsn.name == com_list[i])
    df_tab3_iter=df_tab3_iter.withColumn('words',split(col('text'),' ')).withColumn('word',explode(col('words'))).drop('text','words').groupBy('word').agg(count('word').alias('count')).orderBy('count',ascending=False)
    df_tab3_iter.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+str(i)+"tab4.csv")

#Tab7
df_re_usr = df_user.join(df_review_neg, on="user_id", how="inner")
df_neg_usr = df_re_usr.groupBy("user_id").count().orderBy("count", ascending = False).withColumnRenamed("count","num_neg_re")
df_re_all_usr = df_review.join(df_user, on="user_id", how="inner")
df_all_usr = df_re_all_usr.groupBy("user_id").count().orderBy("count", ascending = False).withColumnRenamed("count","num_all_re")
df_re_user = df_all_usr.join(df_neg_usr, on = "user_id", how = "inner")
df_tab7 = df_re_user.withColumn("percent", df_re_user.num_neg_re/df_re_user.num_all_re*100.0).orderBy("num_neg_re",ascending = False)

#df_tab7.show(10, False)
df_tab7.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"tab7.csv")

user_list = ["ELcQDlf69kb-ihJfxZyL0A","CxDOIDnH8gp9KXzpBHJYXw","bLbSNkLggFnqwNNzzq-Ijw","ic-tyi1jElL_umxZVh8KNA","v7FPnMzdbl6J7U_8H1BWZA","0ygWZ_gXF8qTm0bY95JJqA","gwIqbXEXijQNgdESVc07hg","HEvyblFw4I-UsMqgPGYY_Q","DPitNu466172os6m0Yri1Q","m35ODBrr76JYr20nxoMg_w"]
df_10_usr = df_re_usr#filter(df_re_bsn.name.isin(com_list)).select("text")

from pyspark.sql.functions import split,col,explode,count

df_tab8=df_10_usr.withColumn('words',split(col('text'),' '))\
.withColumn('word',explode(col('words')))\
.drop('text','words').groupBy('word').agg(count('word')\
 .alias('count')).orderBy('count',ascending=False)

df_tab8.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+"tab8.csv")

for i in range(10):
    df_tab9_iter=df_10_usr.filter(df_10_usr.user_id == user_list[i])
    df_tab9_iter=df_tab9_iter.withColumn('words',split(col('text'),' ')).withColumn('word',explode(col('words'))).drop('text','words').groupBy('word').agg(count('word').alias('count')).orderBy('count',ascending=False)
    df_tab9_iter.coalesce(1).write.mode('overwrite').option("header", "true").format('com.databricks.spark.csv').save(output_path+str(i)+"tab9.csv")

