
// Practica 1 - Numero Primo
var _instrucction = "Insert a number and press ENTER..."
println(_instrucction)
var _digit = scala.io.StdIn.readLine().toInt

var _foo = _digit % _digit
var _bar = _digit % 1 

if ( (_foo == 0) && (_bar == 0)  ){
   println("Numero Primo") 
}else {
    println("Numero NO primo")
}
