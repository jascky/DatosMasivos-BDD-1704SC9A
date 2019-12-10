# Unit IV

# Introduction
The present investigation consists in the comparison of three data classification algorithms, in which they will be tested with the same dataset which will help to denote the differences between the algorithms. The algorithms to be compared are decision tree, logistic regression and multilayer perceptron, which are well known and highly used algorithms in the field of data science to binaryly classify large amounts of data. The objective of this research is to find and describe notable differences which could help select one algorithm over another in certain use cases, in which the selected algorithm can provide an improvement in that particular use case. For the comparison, the machine learning framework of the Apache Spark library will be used using the Scala programming language.

# Metrics
### Execution time
For this, the average of several executions of the same algorithm will be calculated to obtain an average execution time of the algorithm, this because in large amounts of data the time it takes to process a dataset can be something key and not necessarily that it has an impact on a better result

### Memery allocated
For this, the total memory used by the algorithm will be measured throughout its execution, the global runtime object that Scala provides in which we can access how much memory was used after executing the program.

```scala
val mb = 1024*1024
val runtime = Runtime.getRuntime
println("Used Memory:  " + (runtime.totalMemory - runtime.freeMemory) / mb + "mb")
```

### Accuracy
The percentage of accuracy of the algorithm will be measured as to whether classification, this will be done by dividing the dataset by 70% for training and another 30% for tests, with this the error coefficient of each algorithm can be determined, it should be clarified that Each algorithm uses different functions to determine the error.

# Results

| Algorithm           | avg exec time | avg memmory usage | Accuracy |
|---------------------|---------------|-------------------|----------|
| Arbol de desicion   | 15.61 s       | 605 mb            | 0.8926   |
| MLP                 | 19.11 s       | 434 mb            | 0.8848   |
| Regresión Logística | 12.18 s       | 324 mb            | 0.8848   |

# Conclusion
According to the results obtained, it can be concluded that there is no exponential or drastically notable difference according to the results with the tests performed with the Bank-Marketing dataset, however, several differences between the algorithms can still be denoted. In the time metric, it can be noted that the logistic regression algorithm has the lowest average execution time. In the memory metric used, it can be seen that logistic regression also takes the lowest average of crazy memory in execution. On the other hand it can be pointed out that in the precision metric decision trees had a higher average although it was the one that used the most memory. Therefore, in this specific case, with the dataset used, it can be said that logistic regression would be a good choice to classify Bank-MArketing dataset data.
