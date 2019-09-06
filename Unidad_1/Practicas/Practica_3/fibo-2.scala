
import scala.math.sqrt
import scala.math.pow

def fib(n: Double): Double = {
    if(n < 2){
        return n
    }
    else {
        var i = ( (1 + sqrt(5)) / 2)
        var j = ( (pow(i,n) - pow((1 - i),n)) / (sqrt(5)))
        return j
    }
    
}

println(fib(10))
