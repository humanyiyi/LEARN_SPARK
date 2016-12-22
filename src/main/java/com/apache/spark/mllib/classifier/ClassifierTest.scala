package com.apache.spark.mllib.classifier


import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/20.
  */
object ClassifierTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("Classifier Test")
    val sc = new SparkContext(conf)

    val rowData = sc.textFile(args(0))
    val records = rowData.map(line => line.split("\t"))
    val data = records.map{ r =>
    val trimmed = r.map(_.replaceAll("\"",""))
    val label = trimmed(r.size - 1).toInt
    val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?" ) 0.0 else d.toDouble)
    LabeledPoint(label, Vectors.dense(features))
    }

    data.cache()
    val nbdata = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"",""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?" ) 0.0 else d.toDouble)
        .map(d => if (d < 0) 0.0 else d)  //朴素贝叶斯模型要求特征值非负。将负特征值设为0
      LabeledPoint(label, Vectors.dense(features))
    }
//    println(data.count())  //result = 7395

    val numIterations = 10
    val maxTreeDepth = 5

    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)  //逻辑回归模型
//    println(lrModel) //org.apache.spark.mllib.classification.LogisticRegressionModel: intercept = 0.0, numFeatures = 22, numClasses = 2, threshold = 0.5
    val svmModel = SVMWithSGD.train(data, numIterations)  //SVM 模型
//    println(svmModel)  //org.apache.spark.mllib.classification.SVMModel: intercept = 0.0, numFeatures = 22, numClasses = 2, threshold = 0.0
    val nbModel = NaiveBayes.train(nbdata)   //朴素贝叶斯模型
//    println(nbModel)     //org.apache.spark.mllib.classification.NaiveBayesModel@345cbf40

    val dtModel = DecisionTree.train(data,Algo.Classification, Entropy, maxTreeDepth)//决策树
//    println(dtModel)
    /**  init: 0.828713267
    total: 1.378146026
    findSplits: 0.756653817
    findBestSplits: 0.515740837
    chooseSplits: 0.50798963
    DecisionTreeModel classifier of depth 5 with 61 nodes*/
//    val dataPoint = data.first
//    val prediction = lrModel.predict(dataPoint.features)
//    println(prediction)  //预测结果1.0
//    val trueLabel = dataPoint.label
//    println(trueLabel)  //实际结果0.0，预测出错

//    val predictions = lrModel.predict(data.map(lp => lp.features))
//    predictions.take(5).foreach(println)  //数据集进行预测

    val lrTotalCorrect = data.map{ point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0 }.sum
    val lrAccuracy = lrTotalCorrect / data.count
//    println(lrAccuracy)  //逻辑回归模型正确率 0.5146720757268425  结果并不好

    val svmTotalCorrect = data.map{ point =>
    if (svmModel.predict(point.features) == point.label) 1 else 0}.sum
    val svmAccuracy = svmTotalCorrect / data.count
//    println(svmAccuracy)  //SVM模型正确率 0.5146720757268425
    val nbTotalCorrect = nbdata.map{ point =>
    if (nbModel.predict(point.features) == point.label) 1 else 0}.sum
    val nbAccuracy = nbTotalCorrect / data.count
//    println(nbAccuracy) //朴素贝叶斯模型正确率 0.5803921568627451

    val dtTotalCorrect = data.map{ point =>
    val score = dtModel.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    if (predicted == point.label) 1 else 0}.sum
    val dtAccuracy = dtTotalCorrect / data.count
//    println(dtAccuracy)  //决策树模型正确率 0.6482758620689655

    val nbMetrics = Seq(nbModel).map{ model =>
    val scoreAndLabels = nbdata.map{ point =>
    val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)}
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
//    println(nbMetrics)  //List((NaiveBayesModel,0.6808510815151734,0.5835585110136261))
    val metrics = Seq(lrModel, svmModel).map{ model =>
      val scoreAndLabels = data.map{ point=>
        (model.predict(point.features),point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
//    println(metrics)  //List((LogisticRegressionModel,0.7567586293858841,0.5014181143280931), (SVMModel,0.7567586293858841,0.5014181143280931))
    val dtMetrics = Seq(dtModel).map{ model =>
      val scoreAndLabels = data.map{ point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR,metrics.areaUnderROC)
    }
    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
//    allMetrics.foreach{ case (m, pr, roc) =>
//      println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
//      /** LogisticRegressionModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
//          SVMModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
//          NaiveBayesModel, Area under PR: 68.0851%, Area under ROC: 58.3559%
//          DecisionTreeModel, Area under PR: 74.3081%, Area under ROC: 64.8837%*/
//    }

    val vectors = data.map(lp => lp.features)
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //计算每列的汇总统计
    /**columnSimilarities(threshold: Double ): CoordinateMatrix 计算每列之间的相识度，采用抽样方法进行计算，参数是threshold  如：matrix.columnSimilarities(0.5)
      * columnSimilarities(): CoordinateMatrix 计算每列之间的相似度
      * computeColumnSummaryStatistics():MultivariateStatisticalSummary  计算每列的汇总统计
      * computeCovariance():Matrix 计算每列之间的协方差，生成协方差矩阵
      * */
//    println(matrixSummary.mean)
//    println(matrixSummary.min)
//    println(matrixSummary.max)
//    println(matrixSummary.variance)
//    println(matrixSummary.numNonzeros)

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaleData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
//    println(scaleData.first.features)  //标准化后的特征向量

      //逻辑回归
    val lrModelScaled = LogisticRegressionWithSGD.train(scaleData, numIterations)
    val lrTotalCorrectScaled = scaleData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaled = lrTotalCorrectScaled / data.count
    val lrPredictionsVsTure = scaleData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTure)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
//    println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%" )
    /** 特征标准化后的结果LogisticRegressionModel
        Accuracy: 62.0419%
        Area under PR: 72.7254%
        Area under ROC: 61.9663%
       */
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
//    println(categories)
    //Map("weather" -> 0, "sports" -> 1, "unknown" -> 10, "computer_internet" -> 11, "?" -> 8, "culture_politics" -> 9, "religion" -> 4, "recreation" -> 7, "arts_entertainment" -> 5, "health" -> 12, "law_crime" -> 6, "gaming" -> 13, "business" -> 2, "science_technology" -> 3)
//    println(numCategories) //14
    val dataCategories = records.map { r=>
      val trimmed = r.map(_.replaceAll("\"",""))
      val  label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
//    println(dataCategories.first)
    //(0.0,[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,
    // 0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])

    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
//    println(scaledDataCats.first.features)  //标准化之后的数据特征
    // [-0.02326210589837061,-0.23272797709480803,2.7207366564548514,-0.2016540523193296,-0.09914991930875496,
    // -0.38181322324318134,-0.06487757239262681,-0.4464212047941535,-0.6807527904251456,-0.22052688457880879,
    // -0.028494000387023734,-0.20418221057887365,-0.2709990696925828,-0.10189469097220732,1.1376473364976751,
    // -0.08193557169294784,1.0251398128933333,-0.05586356442541853,-0.4688932531289351,-0.35430532630793654,
    // -0.3175352172363122,0.3384507982396541,0.0,0.8288221733153222,-0.14726894334628504,0.22963982357812907,
    // -0.14162596909880876,0.7902380499177364,0.7171947294529865,-0.29799681649642484,-0.2034625779299476,
    // -0.03296720969690467,-0.04878112975579767,0.9400699751165406,-0.10869848852526329,-0.27882078231369967]
    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / data.count
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features),point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")
    /**  逻辑回归模型添加类别特征对性能的影响 LogisticRegressionModel
          Accuracy: 66.5720%
          Area under PR: 75.7964%
          Area under ROC: 66.5483%*/
     //朴素贝叶斯模型
    val dataNB = records.map{ r =>
      val trimmed = r.map(_.replaceAll("\"",""))
       val label = trimmed(r.size - 1).toInt
       val categoryIdx = categories(r(3))
       val categoryFeatures = Array.ofDim[Double](numCategories)
       categoryFeatures(categoryIdx) = 1.0
       LabeledPoint(label, Vectors.dense(categoryFeatures))
     }
    val nbModelCats = NaiveBayes.train(dataNB)
    val nbTotalCorrectCats = dataNB.map { point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / data.count
    val nbPredictionsVsTrueCats = dataNB.map { point =>
      (nbModelCats.predict(point.features), point.label)
    }

    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    val nbPrCats = nbMetricsCats.areaUnderPR
    val nbRocCats = nbMetricsCats.areaUnderROC
    println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100.0}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC : ${nbRocCats * 100}%2.4f%%")
    /**数据格式正确后朴素贝叶斯计算结果
      * NaiveBayesModel
      Accuracy: 60.9601%
      Area under PR: 74.0522%
      Area under ROC : 60.5138%
      * */
  }
}
