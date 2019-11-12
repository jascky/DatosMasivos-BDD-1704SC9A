# Basic stadistics

### Correllation
Correlation refers to a mutual relationship or association between quantities. In almost any business, it is useful to express one quantity in terms of its relationship with others. For example, sales might increase when the marketing department spends more on TV ads. Spark MLLib provides a `correlation` function that easily helps calculate the correlation between to quantities.

### Hypothesis testing
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not. Currently Spark MLlib provides hypothesis testing with `chi square test`.

A chi-square test is used in statistics to test the independence of two events. Given the data of two variables, we can get observed count O and expected count E. Chi-Square measures how expected count E and observed count O deviates each other.

### Summsrizer

Spark MLlib offers a easy to use set of functions that can handle operation on dataframes columns one of them is the set of summarizer function that let you sum diferent columns and set of colums wich is pretty useful when working with basics stadisctics.   