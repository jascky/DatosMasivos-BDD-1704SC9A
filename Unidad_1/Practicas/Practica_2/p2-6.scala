// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
//     "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
// 6 a . Imprime todas la llaves del mapa
// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)

var mutmap = collection.mutable.Map( ("jose",20),("luis",24),("Ana",23),("susana",27))
println(mutmap.keys)
mutmap += ("Miguel" -> 23)
println(mutmap)

