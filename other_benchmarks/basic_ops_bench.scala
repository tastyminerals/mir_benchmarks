import util.Random.nextInt

class BenchmarkOps {
  final val rows = 5000
  final val cols = 6000

  def getRandomIntMatrix: Array[Array[Int]] =
    Array.fill(rows, cols) {
      nextInt(10)
    }

  def simple2DSum(
      arr1: Array[Array[Int]],
      arr2: Array[Array[Int]]
  ): Array[Array[Int]] = {
    arr1.flatten.zip(arr2.flatten).map { case (x, y) => x + y }.grouped(cols)
  }

}

object BenchmarkOps {
  def main(args: Seq[String]): Unit = {
    println("OK")
  }
}
