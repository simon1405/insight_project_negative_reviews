#!/usr/bin/env python
# coding: utf-8


import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('S3CSVpro').getOrCreate()

import pandas as pd
import numpy as np
#import nltk

#nltk.download('vader_lexicon',halt_on_error=False)
#from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from pyspark.sql.functions import *

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
#import logging


if __name__ == "__main__": 
    
    df_re = spark.read.json("s3://cf-templates-efn3brqcqavf-us-east-2/yelp/yelp_academic_dataset_review.json")
    df_re = df_re.repartition(50)
    #df_re = df_re.withColumn('vsa_neg', siaUDF_neg(df_re['text']))
    #df_re = df_re.withColumn('vsa_neu', siaUDF_neu(df_re['text']))
    #df_re = df_re.withColumn('vsa_pos', siaUDF_pos(df_re['text']))
    #df_re = df_re.withColumn('vsa_compound', siaUDF_compound(df_re['text']))
    df_c = df_re.count()
    print(df_c)
    #filename = "s3://cf-templates-efn3brqcqavf-us-east-2/yelp/yelp_academic_dataset_review_w.csv"
    #df_re.repartition(1).write.option("sep","|").option("header","true").csv(filename)
