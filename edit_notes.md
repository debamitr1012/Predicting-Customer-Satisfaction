In the original project three different classification methods have been used. Building upon that, four more methods have been implemented with improved in performance. Further,
other metrics like f1 score and jaccard score have also implemented for analysis of performance. <br><br>
See <i>edit_main.py</i> for changes. <br><br>
In the edit, seven different methods have been using for classification:
<ul>
  <li>Decision Tree Classification</li>
  <li>Random Forest Classification</li>
  <li>Gradient Boosting Classification</li>
  <li>Extra Trees Classification</li>
  <li>Bagging Classification</li>
  <li>Adaptive Boost Classification with a DecisionTreeClassifier as the base estimator</li>
  <li>Adaptive Boost Classification with a RandomForestClassifier as the base estimator</li>
</ul>
<br>
<b>The results have been presented: </b><br>


|*Training Data*| Decision Tree | Random Forest | Gradient Boost | Extra Trees | Bagging Classifier | Adaptive Boost D | Adaptive Boost R |
| ------------- | ------------- | ------------- | -------------- | ----------- | ------------------ | ---------------- | ---------------- |
| Accuracy      | 0.871425127   | 0.87012298    | 0.877525922    | 0.851977333 | 0.998794309        | 0.883349409      | 0.887834579      |
| F1 Score      | 0.928212723   | 0.927430004   | 0.931319387    | 0.918715282 | 0.999281578        | 0.93379047       | 0.936760317      |
| Jaccard Score | 0.866041931   | 0.864680167   | 0.871466532    | 0.849651591 | 0.998564188        | 0.875803905      | 0.881043412      |
|               |               |               |                |             |                    |                  |                  |
| *Test Data*   | Decision Tree | Random Forest | Gradient Boost | Extra Trees | Bagging Classifier | Adaptive Boost D | Adaptive Boost R |
| Accuracy      | 0.86727755    | 0.866409453   | 0.870894623    | 0.851073065 | 0.907595852        | 0.870653484      | 0.877212443      |
| F1 Score      | 0.925949844   | 0.925397253   | 0.927701407    | 0.918168327 | 0.946910502        | 0.926737325      | 0.93097278       |
| Jaccard Score | 0.862110432   | 0.861152882   | 0.865152126    | 0.848716441 | 0.899173815        | 0.863476712      | 0.870859751      |



<br>
<i>Note: Adaptive Boost D stands for Adaptive Boost Classification with a DecisionTreeClassifier as the base estimator <br>
         Adaptive Boost R stands for Adaptive Boost Classification with a RandomForestClassifier as the base estimator</i>
<br> <br>       
<b>Immediate Conclusions:</b> <br><br>
<i>Bagging Classifier seems to work the best with a 0.907595852423438 accuracy over the testing data and a 0.998794309139136 accuracy over the training data</i>
<br><br> Note: See <i>Customer Seg Performance Analysis.xlsx</i> for results

