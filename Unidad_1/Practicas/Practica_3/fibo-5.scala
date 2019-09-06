
def fib(n:Int):Int = {

    if (n < 2){
        return n
    }
    else {
        var a = new Array[Int](n + 1)
        a(0) = 0
        a(1) = 1

        for( i <- Range(2, n+1 ) ){
            a(i) = a(i - 1) + a(i-2)
        }
        return a(n)
    }
    
}

println(fib(10)) 