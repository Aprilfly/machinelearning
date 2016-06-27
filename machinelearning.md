### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement-a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is to quantify how much of a particular activity they do,
but they rarely quantify how well they do it. In this project, the goal
is be to use data from accelerometers on the belt,forearm,arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways.

### The Project Goal

The goal of project is to predict manner in which they did the exercise,
which is the "classe" variable in the training set, with any of the
variables in the dataset.Finally use the prediction model to predict 20
different test cases.

### Data Source

The training data for this project are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>  
The test data are available here:  
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

### Load and Read data

Load and read data from links, in which "NA", "" and "\#DIV/0!" are
interpreted as missing values.

    traindata<-fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",na.strings=c("NA","","#DIV/0!"))
    testdata<-fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",na.strings=c("NA","","#DIV/0!"))

### Data Preprocess

Split the Training Datasets into training set and testing set.

    set.seed(1000)
    inTrain<-createDataPartition(y=traindata$classe,p=0.60,list=FALSE)
    training<-traindata[inTrain,]
    testing<-traindata[-inTrain,]

    dim(training) 

    ## [1] 11776   160

    dim(testing)

    ## [1] 7846  160

Each datasets consists of 160 variables.  
Eliminate the variables with excessive missing values and that in 1st 7
columns which is not exercise parameters.

    VarNA<-apply(training,2, function (x) length(which(is.na(x))))
    VarNA<-as.data.frame(VarNA)
    ##Obtain the column names with no missing value
    Var<-row.names(apply(VarNA,2,function(x) x[x==0]))
    ##Eliminate the 1st 7 columns which has nothing to do the predicted variable 'classe'
    Var<-Var[-c(1:7)]

There are 52 variables to be used in prediction and 'classe' as the
outcome.

    Var

    ##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
    ##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
    ##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
    ## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
    ## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
    ## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
    ## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
    ## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
    ## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
    ## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
    ## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
    ## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
    ## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
    ## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
    ## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
    ## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    ## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    ## [52] "magnet_forearm_z"     "classe"

Select the variables used in prediction in training and testing data.

    training<-as.data.frame(training)
    mytraining<-training[which(colnames(training)%in% Var)]
    testing<-as.data.frame(testing)
    mytesting<-testing[which(colnames(testing)%in% Var)]

Remove zero covariates in the mytraining and mytesting dataset.

    nsv1<-nearZeroVar(mytraining,saveMetrics=TRUE)
    ntraining<-mytraining[,nsv1$zeroVar==FALSE]
    nsv2<-nearZeroVar(mytesting,saveMetrics=TRUE)
    ntesting<-mytesting[,nsv2$zeroVar==FALSE]

### Predict with Recursive Partitioning and Regression Trees model

Fit the model with 'classe' as the outcome and all the remaining
varaibles as predictors.

    set.seed(1000)
    modfit<-train(classe~.,method="rpart",data=ntraining)

The accuracy of the model is only 0.5878310. The probability of
predicting 20 test datasets with correct results would only reaches
0.5878310^20=2.426888e-05, which is almost useless in prediction.

    modfit

    ## CART 
    ## 
    ## 11776 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa     
    ##   0.02456099  0.5878310  0.47634203
    ##   0.04347413  0.4661506  0.29167459
    ##   0.11687233  0.3285869  0.06472165
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.02456099.

The root, nodes, split and possibility of being in each class for each
split is presented below for finalModel.

    modfit$finalModel

    ## n= 11776 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##   1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
    ##     2) roll_belt< 129.5 10707 7401 A (0.31 0.21 0.19 0.18 0.11)  
    ##       4) pitch_forearm< -33.15 961    9 A (0.99 0.0094 0 0 0) *
    ##       5) pitch_forearm>=-33.15 9746 7392 A (0.24 0.23 0.21 0.2 0.12)  
    ##        10) yaw_belt>=169.5 504   52 A (0.9 0.054 0 0.042 0.0079) *
    ##        11) yaw_belt< 169.5 9242 6999 B (0.21 0.24 0.22 0.21 0.12)  
    ##          22) magnet_dumbbell_z< -93.5 1128  456 A (0.6 0.28 0.051 0.051 0.023) *
    ##          23) magnet_dumbbell_z>=-93.5 8114 6117 C (0.15 0.24 0.25 0.23 0.14)  
    ##            46) roll_dumbbell< -64.76216 1234  513 C (0.14 0.15 0.58 0.04 0.077) *
    ##            47) roll_dumbbell>=-64.76216 6880 5078 D (0.15 0.25 0.19 0.26 0.15)  
    ##              94) magnet_dumbbell_y>=317.5 3430 2184 B (0.13 0.36 0.053 0.31 0.15)  
    ##               188) total_accel_dumbbell>=5.5 2637 1452 B (0.11 0.45 0.068 0.19 0.18) *
    ##               189) total_accel_dumbbell< 5.5 793  246 D (0.17 0.077 0.0013 0.69 0.061) *
    ##              95) magnet_dumbbell_y< 317.5 3450 2355 C (0.18 0.14 0.32 0.22 0.14) *
    ##     3) roll_belt>=129.5 1069   42 E (0.039 0 0 0 0.96) *

    plot(modfit$finalModel,uniform=TRUE,main="Classification Tree")
    text(modfit$finalModel,use.n=TRUE,all=TRUE,cex=.8)

![](machinelearning_files/figure-markdown_strict/unnamed-chunk-10-1.png)

    testpredict<-predict(modfit,ntesting)
    confusionMatrix(testpredict,ntesting$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1330  239   32   73   17
    ##          B  196  767  113  343  331
    ##          C  563  463 1223  529  394
    ##          D  111   49    0  341   28
    ##          E   32    0    0    0  672
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5523          
    ##                  95% CI : (0.5412, 0.5633)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.4386          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.5959  0.50527   0.8940  0.26516  0.46602
    ## Specificity            0.9357  0.84466   0.6991  0.97134  0.99500
    ## Pos Pred Value         0.7865  0.43829   0.3856  0.64461  0.95455
    ## Neg Pred Value         0.8535  0.87680   0.9690  0.87085  0.89219
    ## Prevalence             0.2845  0.19347   0.1744  0.16391  0.18379
    ## Detection Rate         0.1695  0.09776   0.1559  0.04346  0.08565
    ## Detection Prevalence   0.2155  0.22304   0.4043  0.06742  0.08973
    ## Balanced Accuracy      0.7658  0.67496   0.7966  0.61825  0.73051

From the confusion Matrix, the accuracy of prediction only reaches
0.5523.  
The In Sample Error is 1- 0.5878310=0.412169.  
The Out Sample Error is 1- 0.5523=0.4477.

### Predict with Random Forest model

The algorithm of Random Forest is a time-consuming method, which propels
us to use parallel processing. But the tradeoff made in this analysis is
changing the resampling method from the default of bootstrapping to
k-fold cross-validation. The change in resampling technique may trade
processing performance for reduced model accuracy. However experiment
indicates that 5 fold cross-validation resampling technique delivered
the same accuracy as the more computationally expensive bootstrapping
technique. Here we use 10 fold cross-validation resampling.

The process for executing the random forest model parallely is as
follows.  
1- Configure parallel processing  
2- Configure trainControl object  
3- Develop training model  
4- De-register parallel processing cluster

    set.seed(1000)
    ##Configure parallel processing
    cluster<-makeCluster(detectCores()-1)
    registerDoParallel(cluster)
    ##Configure trainControl object
    fitControl<-trainControl(method="cv",number=10,allowParallel=TRUE)
    modfit2<-train(classe~.,method="rf",data=ntraining,trControl=fitControl)
    ##De-register parallel processing cluster
    stopCluster(cluster)
    modfit2

    ## Random Forest 
    ## 
    ## 11776 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 10598, 10597, 10598, 10600, 10598, 10598, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9902344  0.9876457
    ##   27    0.9911687  0.9888288
    ##   52    0.9880266  0.9848536
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

The accuracy of the model reaches 0.9912.  
The In Sample Error is 1- 0.9912=0.0088.

    testpredict2<-predict(modfit2,ntesting)
    confusionMatrix(testpredict2,ntesting$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2229   24    0    0    0
    ##          B    3 1492   12    1    2
    ##          C    0    2 1351   24    2
    ##          D    0    0    5 1261    5
    ##          E    0    0    0    0 1433
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9898          
    ##                  95% CI : (0.9873, 0.9919)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9871          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9987   0.9829   0.9876   0.9806   0.9938
    ## Specificity            0.9957   0.9972   0.9957   0.9985   1.0000
    ## Pos Pred Value         0.9893   0.9881   0.9797   0.9921   1.0000
    ## Neg Pred Value         0.9995   0.9959   0.9974   0.9962   0.9986
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2841   0.1902   0.1722   0.1607   0.1826
    ## Detection Prevalence   0.2872   0.1925   0.1758   0.1620   0.1826
    ## Balanced Accuracy      0.9972   0.9900   0.9916   0.9895   0.9969

The accuracy of the prediction reaches 0.9898.  
The Out Sample Error is 0.0102.

### Predict the test data

    testresult<-predict(modfit2,newdata=testdata)
    testresult

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
