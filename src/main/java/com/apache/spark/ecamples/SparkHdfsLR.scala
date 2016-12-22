package com.apache.spark.ecamples



import java.util.{Random, StringTokenizer}

import breeze.linalg.{DenseVector, Vector}
import breeze.numerics.exp
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/12/15.
  */
object SparkHdfsLR {
  val D = 10
  val rand = new Random()

  case class DataPoint(x: Vector[Double], y: Double)

  def parsePoint(line: String): DataPoint ={
    val tok = new StringTokenizer(line, " ")
    var y = tok.nextToken.toDouble
    var x = new Array[Double](D)
    var i = 0
    while (i < D) {
      x(i) = tok.nextToken.toDouble; i += 1
    }
    DataPoint(new DenseVector(x), y)
  }

  def showWarning(): Unit ={
    System.err.println(
      """WARN: This is a naive implementation of Logistic Regression and is given as an example!
        |Please use org.apache.spark.ml.classification.LogisticRegression
        |for more conventional use.
      """.stripMargin)
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.out.println("Usage: SparkHdfsLR <file> <iters>")
      System.exit(1)
    }

    showWarning()

    val conf = new SparkConf().setMaster("local").setAppName("SparkHdfsLR")
    val sc = new SparkContext(conf)

    val inputPath = args(0)
    val lines = sc.textFile(inputPath)

     val points = lines.map(parsePoint).cache()

    val ITERATIONS = args(1).toInt

    var w = DenseVector.fill(D) {2 * rand.nextDouble - 1}
    println("Initial w: " + w)

    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      val gradient = points.map { p =>
        p.x * (1 / (1 + exp(-p.y * (w.dot(p.x)))) - 1) * p.y
      }.reduce(_ + _)
      w -= gradient
    }
    println("Final w: " + w)
    sc.stop()
  }
}
