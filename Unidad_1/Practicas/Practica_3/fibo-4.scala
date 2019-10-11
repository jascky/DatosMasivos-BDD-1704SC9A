
def fibo(n:Int):Int = {
    var a = 0
    var b = 1

    for(i <- Range(0,n)){
        b = b + a
        a = b - a
          
    }
    return a
}

println(fibo(10))
