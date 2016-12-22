//ALS是交替最小二乘法，通常用在推荐系统算法中

package com.apache.spark.ecamples

import org.apache.commons.math3.linear._

/**
  * Created by root on 2016/12/5.
  */
object LocalALS {

  var M = 0
  var U = 0
  var F = 0
  var ITERATIONS = 0
  val LAMBDA = 0.01

  def randomMatrix(rows: Int, cols: Int): RealMatrix =
    new Array2DRowRealMatrix(Array.fill(rows, cols)(math.random))

  def generateR():RealMatrix = {
    val mh = randomMatrix(M, F)
    val uh = randomMatrix(U, F)
    mh.multiply(uh.transpose())
  }
  def rmse(targetR: RealMatrix, ms: Array[RealVector], us: Array[RealVector]): Double = {
    val r = new Array2DRowRealMatrix(M, U)
    for (i <- 0 until M; j <- 0 until U){
      r.setEntry(i, j, ms(i).dotProduct(us(j)))
    }
    val diffs = r.subtract(targetR)
    var sumSqs = 0.0
    for (i <- 0 until M; j <- 0 until U){
      val diff = diffs.getEntry(i, j)
      sumSqs += diff* diff
    }
    math.sqrt(sumSqs / (M.toDouble * U.toDouble))
  }

   def upDateMovie(i: Int, m: RealVector, us : Array[RealVector], R: RealMatrix): RealVector = {
     var Xtx:RealMatrix = new Array2DRowRealMatrix(F, F)
     var Xty: ArrayRealVector = new ArrayRealVector(F)
     for (j <- 0 until U){
       val u = us(j)
       Xtx = Xtx.add(u.outerProduct(u))
       Xty = Xty.add(u.mapMultiply(R.getEntry(i, j)))

     }
     for (d <- 0 until F){
       Xtx.addToEntry(d, d, LAMBDA * U)
       }
     new CholeskyDecomposition(Xtx).getSolver.solve(Xty)
   }

  def updateUser(j: Int, u: RealVector, ms: Array[RealVector], R: RealMatrix) :RealVector = {
    var Xtx: RealMatrix = new Array2DRowRealMatrix(F, F)
    var Xty: RealVector = new ArrayRealVector(F)
    for (i <- 0 until M){
      val m = ms(i)
      Xtx = Xtx.add(m.outerProduct(m))
      Xty = Xty.add(m.mapMultiply(R.getEntry(i, j)))

    }
    for (d <- 0 until F) {
      Xtx.addToEntry(d, d, LAMBDA * M)
    }
    new CholeskyDecomposition(Xtx).getSolver.solve(Xty)
  }

  def showWarning(): Unit ={
    System.err.println(
      """WARN: This is a naive implementation of ALS and is given as an exaple!
        |Please use org.apache.spark.ml.recommendation.ALS
        |for more conventional use.
      """.stripMargin
    )
  }

  def randomVector(n: Int): RealVector =
    new ArrayRealVector(Array.fill(n)(math.random))

  def main(args: Array[String]): Unit = {
    args match{
      case Array(m, u, f, iters) =>
        M = m.toInt
        U = u.toInt
        F = f.toInt
        ITERATIONS = iters.toInt
      case _ =>
        System.err.println("Usage: LocalALS <M> <U> <F> <iters>")
        System.exit(1)

    }
    showWarning()
    println(s"Running with M=$M, U=$U, F=$F, iters=$ITERATIONS")
    val R = generateR()

    var ms = Array.fill(M)(randomVector(F))
    var us = Array.fill(U)(randomVector(F))

    for (iter <- 1 to ITERATIONS) {
      println(s"Iteration $iter:")
      ms = (0 until M).map(i => upDateMovie(i, ms(i), us, R)).toArray
      us = (0 until U).map(j => updateUser(j, us(j), ms, R)).toArray
      println("RMSE = " + rmse(R, ms, us))
      println()
    }
  }
}
