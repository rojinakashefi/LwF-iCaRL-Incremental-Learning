# ResNet Incremental Learning

## Dataset

1. Used CIFAR100 dataset

2. Since we are using incremental learning we need to modify the dataset class in a way which each time we get data for specific number of classes (num_classes). For example in our continual learning in each iteration the model is learning 10 new classes to the network in this way num_classes would be 10 and will take data and labels of only 10 classes from 100 classes.

## ResNet Architecture

1. Resnet consists of basic blocks and bottleneck blocks. (Implementation of both of these classes)
   
   <img src="data:image/png;base64,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" title="" alt="" width="557">

2. Creating different resnet architectures by defining what block to use, how many block and block outputs size for example resnet 32 uses basic block each one has 5 blocks and the outputs size are 16,32,64.

3. Implementation on both ResNet 32, 32, 64.

## Finetuning:

1. Define ResNet32 architecture with 100 output classes.

2. Each batch consists of data of 10 classes. (batch_size = 128) . The model has been trained 10 times. (each time a new 10 classes)

3. Trained each 10 classes for 70 epoch, Tested each 10 classes after 70 epoch.

4. Tested model on **all** classes that has been learned till then.
   
   We can see **catastrophic forgetting** by getting high results in test accuracy in Step3 and low accuracy in Step4.
   
   <img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/finetuning-2.png" alt="" width="428" data-align="center">
   
   In the above picture it is only about last 10 classes, we can see the train loss is so low and also the test loss on only the data of 10 last class is so low. Which means the model learned the 10 last class with high accuracy. (Step 3) However we can see the model has forgotten about the 9th first class in below picture. (Step 4)
   
   <img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/finetuning-1.png" alt="" width="433" data-align="center">
   
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

<img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/LWF" alt="Screenshot 2023-05-25 at 3.27.30 PM.png" width="459" data-align="center">

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

<img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/icarl-1.png" alt="" width="342" data-align="center">

The individual accuracy for batches decreases since they need to classify classes from pervious batches (exemplars).

<img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/icarl-2.png" alt="" width="396" data-align="center">

**Herding classifier:**

In 10'th iteration herding has better results than random exemplars.

<img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/icarl-3.png" alt="icarl-3.png" width="414" data-align="center">

<img title="" src="https://github.com/rojinakashefi/GradCam-ResNet-IncrementalLearning/blob/main/images/icarl-4.png" alt="icarl-4.png" width="418" data-align="center">
