
def fib(n:Int):Int = {
    var a = 0 
    var b = 1
    var c = 0 

    for(i <- Range(0,n)){
        c = b + a 
        a = b
        b = c
    }
    return a
}

println(fib(10))
    