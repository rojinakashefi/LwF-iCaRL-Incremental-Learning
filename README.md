# ResNet Incremental Learning

## Dataset

1. Used CIFAR100 dataset

2. Since we are using incremental learning we need to modify the dataset class in a way which each time we get data for specific number of classes (num_classes). For example in our continual learning in each iteration the model is passing 10 new classes to the network in this way num_classes would be 10 and will take data and labels of only 10 classes from 100 classes.

## ResNet Architecture

1. Resnet consists of basic blocks and bottleneck blocks. (Implementation of both of these classes)
   
   <img src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/resnet.png" title="" alt="" width="557">

2. Creating different resnet architectures by defining what block to use, how many block and block outputs size for example resnet 32 uses basic block each one has 5 blocks and the outputs size are 16,32,64.

3. Implementation on both ResNet 20, 32, 64.

## Finetuning:

1. Define ResNet32 architecture with 100 output classes.

2. Each batch consists of data of 10 classes. (batch_size = 128) . The model has been trained 10 times. (each time a new 10 classes)

3. Trained each 10 classes for 70 epoch, Tested each 10 classes after 70 epoch.

4. Tested model on **all** classes that has been learned till then.
   
   We can see **catastrophic forgetting** by getting high results in test accuracy in Step3 and low accuracy in Step4.
   
   <img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/finetuning-2.png" alt="" width="428" data-align="center">
   
   In the above picture it is only about last 10 classes, we can see the train loss is so low and also the test loss on only the data of 10 last class is so low. Which means the model learned the 10 last class with high accuracy. (Step 3) However we can see the model has forgotten about the 9th first class in below picture. (Step 4)
   
   <img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/finetuning-1.png" alt="" width="433" data-align="center">
   
   In the above picture we can see in the first iteration which we have trained the model on 10 classes we get high accuracy but in the second iteration testing on whole 20 classes resulted on having lower accuracy, which means the model forgot a bit about the first 10 classes, as we continue we can see in 10 iteration the model accuracy is about 0.1 which means it can't classify the 100 learned classes well and has forgotten about 90 first classes. Also the loss increases as we continue.

## Learning without Forgetting

1. One of the Incremental learning techniques to face catastrophic forgetting problem. 

2. Used Knowledge Distillation loss. For learning more about this technique check this [link](https://www.youtube.com/watch?v=gADXP5daZeM&t=321s&ab_channel=DinguSagar).

3. First created resnet32 with 10 output class, on the first 10 class the model is a simple CNN.

4. After the first 10 classes, the new 10 classes is added to the model resnet32 and we have a model with 20 output classes.

5. Compute the loss using :
   
   1. **L1:** Loss between the <u>old model on new data</u> and <u>new model on new data</u> using Multi nomial Logistic Loss.
   
   2. **L2:** Loss between <u>new model on new data</u> and <u>ground truth labels</u> using Cross Entropy.
   
   3. **Total Loss:** Lambda * L1 + L2 (Lambda is a weight)
   
   4. By increasing the value of Lambda the model will try to remeber the old classes more than new classes.

6. Calculate the test accuracy of 10 classes at the end of each epoch.

7. Compute the total test accuracy of the classes which has been trained till then.

<img src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/Lwf.png" width="459" data-align="center">

We can see using this technique in first 20 class we have accuracy around 50 percent however in previous technique the accuracy for first 20 class was near 25 percent.

## iCaRL

In this method we use set of exemplars to remember some data from previous classes.

1. First we update the data by adding  new_data to examplars

2. Then we update the net by adding 10 new layers with random weights

3. In training phase we compute the loss using binary cross entropy between the <u>old network with new data concatenated with labels of new classes (making a target vector)</u> and <u>new_network with new_data</u>.

4. For choosing exemplars there are two ways:
   
   1. Random exemplars: choosing m exmaplers randomly
   
   2. Herding exemplars: Computing the mean of feature vector representation of all classes. Then Minimize the distance between feature representation and the curret class mean.

5. Reduce the exempleres to preserve memory

6. For classification task we use Near Mean exemplars which for each image we associate the label corresponding to the minimal distance to the class mean of each exemplars set.

**Random classifier**:

We can see the accuracy in our 10'th iteration is around 50 percent.

<img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/icarl-1.png" alt="" width="342" data-align="center">

The individual accuracy for batches decreases since they need to classify classes from pervious batches (exemplars).

<img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/icarl-2.png" alt="" width="396" data-align="center">

**Herding classifier:**

In 10'th iteration herding has better results than random exemplars.

<img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/icarl-3.png" alt="icarl-3.png" width="414" data-align="center">

<img title="" src="https://github.com/rojinakashefi/LwF-iCaRL-Incremental-Learning/blob/main/images/icarl-4.png" alt="icarl-4.png" width="418" data-align="center">
