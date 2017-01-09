package com.udbac.scala.test
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by root on 2016/12/30.
  */
object ScalaTest {
  case class RawDataRecord(category: String, text: String)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("scalaTest")
    val sc = new SparkContext(conf)

    val sQLContext = new SQLContext(sc)
    import sQLContext.implicits._
    var srcDF = sc.textFile("D:\\UDBAC\\LEARN_SPARK\\data\\1.txt").map{
      x =>
        var data = x.split(",")
        RawDataRecord(data(0),data(1))
    }.toDF()

    srcDF.select("category", "text").take(2).foreach(println)

    var tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    var wordsData = tokenizer.transform(srcDF)

    wordsData.select($"category", $"text", $"words").take(2).foreach(println)

    var hashIngTF =
      new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(100)
    var featurizedData = hashIngTF.transform(wordsData)

    featurizedData.select($"category",$"words",$"rawFeatures").take(2).foreach(println)
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    var idfModel = idf.fit(featurizedData)
    var rescaledData = idfModel.transform(featurizedData)
    rescaledData.select($"category",$"words",$"features").take(2).foreach(println)
    val test = rescaledData.select($"category",$"features")
    val trainDataRdd = test.map {
      case Row(label: String, features: Vector) =>
        LabeledPoint(label.toDouble, Vectors.dense(features.toArray))
    }
    trainDataRdd.take(1)
  }
}
