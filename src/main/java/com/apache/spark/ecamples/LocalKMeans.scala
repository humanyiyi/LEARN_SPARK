//K-Means算法
package com.apache.spark.ecamples

import breeze.linalg.{squaredDistance, DenseVector, Vector}
import scala.collection.mutable
import java.util.Random


/**
  * Created by root on 2016/12/9.
  */
object LocalKMeans {
  val N = 1000
  val R = 1000
  val D = 10
  val K = 10
  val convergeDist = 0.001
  val rand = new Random(42)

  def generateData: Array[DenseVector[Double]] = {
    def generatePoint(i: Int): DenseVector[Double] = {
      DenseVector.fill(D) {
        rand.nextDouble * R
      }
    }
    Array.tabulate(N)(generatePoint)
  }

  def closestPoint(p: Vector[Double], centers: mutable.HashMap[Int, Vector[Double]]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 1 to centers.size) {
      val vCurr = centers.get(i).get
      val tempDist = squaredDistance(p, vCurr)
      if (closest > tempDist) {
        closest = tempDist
        bestIndex = i
      }
    }
    bestIndex
  }

  def showWarning(): Unit = {
    System.err.println(
      """WARN: This is a naive implementation of KMeans Clustering and  is given as an example!
        |Please use org.apache.spark.ml.clustering.KMeans
        |for more conventional use
      """.stripMargin)
  }

  def main(args: Array[String]) {
    val data = generateData
    var points = new mutable.HashSet[Vector[Double]]
    var kPoints = new mutable.HashMap[Int, Vector[Double]]
    var tempDist = 1.0
    while (points.size < K) {
      points.add(data(rand.nextInt(N)))
    }
    val iter = points.iterator
    for (i <- 1 to points.size) {
      kPoints.put(i, iter.next())
    }

    println("Initial centers: " + kPoints)

    while (tempDist > convergeDist) {
      var closest = data.map(p => (closestPoint(p, kPoints), (p, 1)))
      var mappings = closest.groupBy[Int](x => x._1)
      var pointStats = mappings.map { pair =>
        pair._2.reduceLeft[(Int, (Vector[Double], Int))] {
          case ((id1, (p1, c1)), (id2, (p2, c2))) => (id1, (p1 + p2, c1 + c2))
        }
      }

      var newPoints = pointStats.map { mapping =>
        (mapping._1, mapping._2._1 * (1.0 / mapping._2._2))
      }

      tempDist = 0.0
      for (mapping <- newPoints) {
        tempDist += squaredDistance(kPoints.get(mapping._1).get, mapping._2)
      }
      println(tempDist)

      for (newP <- newPoints) {
        kPoints.put(newP._1, newP._2)
      }
    }

    println("Final centers: " + kPoints)

  }

}
