
import scala.math.pow

def fib(n:Int):Int = {
    if (n <= 0){
        return 0 
    }
    else {
        var i = n -1
        var auxOne = 0
        var auxTwo = 1
        var tup1 = (auxTwo,auxOne)
        var tup2 = (auxOne,auxTwo)
        while( i > 0 ) {

            if ( (i % 2) !=  0 ){
                auxOne = (tup2._2 * tup1._2) + (tup2._1 * tup1._1)
                auxTwo =  (tup2._2 * (tup1._2 + tup1._1)) + (tup2._1 * tup1._2)
                tup1 = (auxOne,auxTwo)
            }
            auxOne = (pow(tup2._1,2) + pow(tup2._2,2) ).toInt
            auxTwo = tup2._2 * ( (2*tup2._1)+ tup2._2 )
            tup2 = (auxOne,auxTwo)
            i = i / 2    
            	
        }
        return tup1._1 + tup1._2
        
    }
}

println(fib(10))
