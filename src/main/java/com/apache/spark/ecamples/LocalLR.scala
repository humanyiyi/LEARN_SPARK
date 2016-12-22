//逻辑回归算法
package com.apache.spark.ecamples

import breeze.linalg.{DenseVector, Vector}

import scala.util.Random

/**
  * Created by root on 2016/12/2.
  */
object LocalLR {
  val N = 10000
  val D = 10
  val R = 0.7
  val ITERATIONS = 5
  val rand = new Random(42)

  case class DataPoint(x: Vector[Double], y: Double)

  def generateDate: Array[DataPoint] = {
    def generatePoint(i: Int): DataPoint = {
      val y = if (i % 2 == 0) -1 else 1
      val x = DenseVector.fill(D) {rand.nextGaussian + y *R}
      DataPoint(x, y)
      }
      Array.tabulate(N)(generatePoint)
  }

  def showWarning(){
    System.out.println(
      """WARN: This is a naive implementation of Logistic Regression and is given as an example!
        |Please use org.apache.spark.ml.classification.LogisticRegression
        |for more conventional use.
      """.stripMargin
    )
  }

  def main(args: Array[String]): Unit = {
    showWarning()

    val data = generateDate
    var w = DenseVector.fill(D){2 * rand.nextDouble - 1}
    println("Intitial w: " + w)

    for (i <- 1 to ITERATIONS) {
      println("On iteration " + i)
      var gradient = DenseVector.zeros[Double](D)
      for (i <- 1 to ITERATIONS){
        println("On iteration " + i)
        var gradient = DenseVector.zeros[Double](D)
        for (p <- data) {
          val scale = (1 / (1 + math.exp(-p.y * (w.dot(p.x)))) - 1) * p.y
          gradient += p.x * scale
        }
        w -= gradient
      }
      println("Filnal w: " + w)
    }
  }
}
