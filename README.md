# Artificial Intelligence and Machine Learning - Group Project
# Customer Segmentation
### Team members: Eleonora Di Mario, Martina Di Segni, Yasmin Bocatto  

## Introduction

The first step to build any model is framing the problem and looking at the big picture. The department wants to segment their customers to develop a targeted email campaign. Our goal is to identify the ideal number of partitions and assign each user in the dataset to one of them, this segmentation has to be done following the RFM analysis.

The RFM analysis stands for Recency, Frequency and Monetary value, each corresponding to key customer traits. These RFM metrics are important indicators of a customer’s behavior. An RFM model is built using three keys factors: 

1.   How recently a customer has transacted with a brand.
2.   How frequently they’ve engaged with a brand.
3.   How much money they’ve spent on a brand’s products and services. 

RFM Analysis enables marketers to increase revenue by targeting specific groups of existing costumers with messages and offers that are more likely to be relevant based on data about a particular set of behaviours. 
To be more precise: 
*   Recency value refers to the amount of time since a costumers last interaction with a brand. This is a key metric because customers who have interacted with your brand more recently are more likely to respond to new marketing efforts. 
*   Frequency value refers to the number of times a customer has made a purchase during a particular period of time. Frequency is a key metric because it shows how deeply a customer is engaged with the brand. High degree of loyalty is measured by greater frequency.
*   Monetary value refers to the total amount of money a customer has spent purchasing products and services from the brand. This one is a key metric because the customers who spent most in the past are more likely to spend in the future.

Considering our objective, just by looking at the dataset we can make some basic but helpful assumptions:
- Some columns have the same informations (product_category_name and product_category_name_english);
- Not everything is fundamental for detecting a pattern in costumer behaviour;
- Since we need to use Recency, Frequency and Monetary values we will look more into the data that is essential for calculating them;
- We are dealing with an unsupervised learning method;


## Methods
For this assignment, the three clustering algorithms that we chose to implement are: k-means, hierarchical clustering and DBSCAN. Even if most clustering algorithms work for costumer segmentation, evaluating the options, for our task specifically, we decided these three options were the best to implement because of some characteristics.

*   K-means clustering is one of the most popular clustering algorithms. The goal of k means is to group data points into distinct non-overlapping subgroups. Customer segmentation is actually one of the major application of this clustering algorithm since it is able to get a better understanding of the segments, which can be used to increase the revenue of the company or in order to adapt different managerial and marketing strategies. K-means also scales to large data sets. 

*   Hierarchical clustering is great when we need to identify the observations that are most similar to a given observation, and the dendrogram that is created can help by identifying the segments and the ideal number of clusters. 

*   DBSCAN is really useful when we are not looking to predict a particular outcome variable, but when we have a set of features we want to use to identify patterns across our dataset. Additionally, DBSCAN is not powerfully influenced by noise or outliers.

But before implementing the models, there were some steps that we performed on the dataset:

1.   Imported the following libraries:

  *   Sklearn
  *   Scipy
  *   Pandas
  *   Matplotlib.pyplot
  *   Numpy
  *   Seaborn

In addition we also imported:
`pd.options.mode.chained_assignment = None  # default='warn'`
`import warnings`
`warnings.simplefilter(action='ignore', category=FutureWarning)`
Since we were getting some warnings because we are using some built-in functions that pandas is going to deprecate, we imported this library in order to have a cleaner output.



2.   Imported the dataset

  *   "customer_segmentation.csv"

3.   Looked at the data structure by using the following functions:

  *   `df.head()`
  *   `df.info()`
  *   `df.describe()`
  *   `df.shape`

4. Performed some data cleaning:

  *   Fixed the date variables by converting them to pandas datetime objects for further analysis
  *   Removed duplicates
  *   Checked for missing values (there were none)
  *   Plotted some scatterplots to look for outliers (found a few in the payment_value variable)
      *  This variable's mean was of 195.2 and it's standard deviation was 295.5, five values were higher than 4000, so we ended up removing them

    ![Payment Value Scatterplot (w/ outliers)](C:/Users/eleon/Documents/GitHub/E12933/payment value scatterplot (with outliers).png)

    ![Payment Value Scatterplot (wo/ outliers)](C:/Users/eleon/Documents/GitHub/E12933/payment value scatterplot (without outliers).png)
  
  *   Grouped similar categories of product_category_name_english such as "drink", "food" and "food_drink" into only "food_drink" to help with future encoding
  *   Encoded the variables customer_state, payment_type, seller_state and product_category_name_english

5. Moved on to exploratory analysis:

  *   Plotted a correlation matrix heatmap to look at the relationship between variables
  
     ![Correlation Matrix Heatmap](C:/Users/eleon/Documents/GitHub/E12933/corr matrix - heat map.png)

  * Plotted some barplots to look at the relationship between categorical and numericall variables, for example, number of customers in each state

    ![Number of Customers per State](C:/Users/eleon/Documents/GitHub/E12933/number of customers per state.png)
    
    ![Cities with more Customers](C:/Users/eleon/Documents/GitHub/E12933/cities with more customers.png)
    
    ![Top Products Ordered X Less Products Ordered](C:/Users/eleon/Documents/GitHub/E12933/top and lowest products.png)

    ![Payment Type and Payment Installments](C:/Users/eleon7Documents/GitHub/E12933/payment type and installments.png)

6.  Started the RFM analysis by calculating the Recency, Frequency and Monetary values:

    * Recency = time since a customer's last purchase
         * For this variable, we calculated each purchasing timestamp minus the most recent purchase timestamp
    * Frequency = how many times has a customer made a purchase
         * Here, we counted purchases per customer_unique_id 
    * Monetary = total amount a customer has spend purchasing products
         * Calculated by summing all the payment_values each customer has spent

7.   Finally, we move on to some RFM exploratory analysis:

    * Created scores for each RFM variable and classified customers considering them
    * Plotted a few graphs to visualize the data

    * Deepened the analysis to set as a baseline for comparison to the models outputs

8.   Standard scaled the data to fit into the algorithms 

After all these steps, we built, applied and validated the models. Additionally we chose the best validated model and interpreted it's clusters putting them in certain categories.

## Experimental Design

We conducted several experiments for the exploratory analysis. The main purpose was to understand each customer data, the correlation between the variables available and the main trends present in our dataset in order to be able to segment customer in the most accurate way for the company to address the specific and targeted marketing campaigns. 
One of the main and most valuable steps was the RFM analysis. RFM stand for Recency, Frequency and Monetary value, each corresponding to key customer traits. RFM was fundamental in our analysis because the three indicators by which it is composed are indicators of customer’s behavior that help predict future behavior. For each of these attributes we assigned a score to each customer, ranging from 1 to 5 and we obtained some interesting results that we plotted:

  ![Recency visualization](C:/Users/eleon/Documents/GitHub/E12933/recency (rfm graphs).png)

We decided to assign a recency score to every costumer according to the date of their last purchase.  We assigned a 5 to people who’s last purchase was in the range of a month, a 4 to people whose last purchase was between 30 and 60 days.  Then we assigned a 3 to people whose last purchase was between 60 and 120 days,  a 2 to customers that purchased their last item in the last 120 to 180 days and lastly a 1 to costumers whose last purchase was over 180 days ago. 
We found out that most of the costumers were assigned a score of 3  while very few costumers received a score of 1.  As we can see the tallest bar is the one that goes from 60 to 120 days, that is the costumers that received a score of 3, that are almost 6000. The bar before that one is the sum of the costumers that obtained a score between 5 and 4, there are around 5500 costumers in this category. Than we have the lowest bar that are the costumers that have a recency score of 1 or 2, and those sum up to 2100. 

  ![Frequency visualization](C:/Users/eleon/Documents/GitHub/E12933/frequency (rfm graphs).png)

We assigned a frequency score to each customer according to the amounts of time they purchased from the company. 
In order for a costumer to have a score of 5 they have to have purchased at least 10 times. Costumers that have purchased between 6 and 9 times (both included) received a score of 4, if a costumer has purchased 4 or 5 times the score assigned was a 3. Lastly if a costumer had only bought from the brand 3 of 2 times, they received a score of 2 and if they only purchased once the score was 1. 
From the graph we can notice how most people bought from this company just once or twice, in fact the amount of people that got a score of 1 are 9536. Very few people (just 58) have obtained a score of 5.

  ![Monetary visualization](C:/Users/eleon/Documents/GitHub/E12933/monetary (rfm graphs).png)

Also for the monetary value each customer was assigned a score according to the total amount spent in the company’s products. We considered costumers that spent more than 500 euros with a score of 5, costumer that spent between 250 and 500 got a score of 4, the costumers that got a 3 were the ones that spent between 150 and 250, the ones that spent between 100 euros and 150 got a rating of 2 and lastly a 1 was assigned to all costumers that spent at most 100 euros. From this resulting graph we can see how most of the customers of the brand spent around 300 euros, while there isn’t a lot of people spending more than 750, and almost no-one spent more than 1100.  In fact most of the costumers received a score of 1 (4957 costumers).

Doing the RFM analysis allowed us to start making a segmentation of customers and use it later on as a baseline to compare the models outputs. At this point, every customer has three scores, each one of them representing the monetary value, the recency and the frequency. In order to do a segmentation we wanted each customer to have only one score, and for this reason we decided to average the scores in order to classify at best each client. However we did not perform an arithmetic average but a weighted one. We assigned a weight equal to two to the frequency score and the monetary value, while to the recency score we assigned a weight of one. This decision was taken after making some considerations about the meaning of each trait. We considered that recency was a less determining factor in our analysis because it’s a score that doesn’t help specifically understand how will a customer behave in the future. Observing our data we noticed that there are a lot of people that only made one purchase, or whose monetary value was not high and this information might not be correlated to how long ago the last purchase was. We also considered that a really frequent client, considered silver or gold, might be penalised too much if the recency score had the same weight as the other two features, also considering that if a customer has a really high monetary value he or she might wait some time before going back to purchase other items.  
We decided to segment customers in three main clusters: Gold, Silver and Bronze.
To the gold segment were assigned the customers having a final score ranging between four and five, the silver cluster are all the customers with a final score between two and four while to the bronze cluster belongs the customer with a score between one and two.  
We then plotted the results using a piechart type of graph.

  ![Piechart customers](C:/Users/eleon/Documents/GitHub/E12933/percentage of customer segments.png)

The results from our clustering were the following:
*  48.8% of the customers were classified as bronze 
*  47.5% of the customers were classified as silver
*  3.7% of the customers were classified as gold 

We can notice how the gold cluster is way smaller than the other two, but we think it’s good to have such a restricted group of gold costumers so the company can target them for more luxury campaigns since those are the costumer that we would consider as an ‘elite’.

With the RFM analysis we were already able to start doing a segmentation for the company but, to further analyse the different clusters, we then implemented some more complicated clustering models such as: K-means, Hierarchical clustering and DBSCAN. 

Moreover, clustering validation has been recognized as one of the important factors essential to the success of clustering algorithms so after implementing the models, this was our next step: choosing the validation metrics. How to effectively and efficiently assess the clustering results of clustering algorithms is the key to the problem. To test how well these models perform and compare them in order to find to most efficient one for this dataset we used some evaluation metrics: Silhouette coefficient, the Calinski-Harabasz index and the Davies Bouldin score.


Silhouette coefficient is a metric to calculate the goodness of a clustering technique. Its values ranges from -1 to 1. A score of one denotes the best meaning that the data points is very compact within the cluster to which it belongs and far away from the other clusters. The worst value is -1. Values near 0 denote overlapping clusters. The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b) . To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of.

The Calinski-Harabasz index is an internal cluster valuation index. It can be used to evaluate the model when ground truth labels are not known where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset. The CH index is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Here cohesion is estimated based on the distances from the maya points in a cluster to its cluster centroid and separation is based on the distance if the cluster centroids from the global centroid. A high CH means better clustering since observations in each cluster are closer together (more dense), while clusters themselves are further away from each other (well separated).

The Davies Bouldin score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between cluster distances. Thus clusters which are farther apart and less dispersed will result in a better score. Davies Bouldin index is calculated as the average similarity of each Cluster (say Ci) to its most similar Cluster (say Cj). It’s an internal evaluation scheme, where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset. Clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.


## Results

After applying the models, we ended up with the following results for each method:

*  K-Means
  *  Number of clusters: 3
  *  Silhouette Score: 0.475
  *  Calinski-Harabasz Score: 9224.910
  *  Davies Bouldin Score: 0.787
         
             
      ![K-means elbow method](C:/Users/eleon/Documents/GitHub/E12933/elbow method.png)   
      ![K-means clusters](C:/Users/eleon/Documents/GitHub/E12933/clusters kmeans.png)

* Hierarchical clustering
  *  Number of clusters: 4
  *  Silhouette Score: 0.388
  *  Calinski-Harabasz Score: 7169.908
  *  Davies Bouldin Score: 0.808

      ![HC dendogram](C:/Users/eleon/Documents/GitHub/E12933/dendogram hc.png) 
      ![HC clusters](C:/Users/eleon/Documents/GitHub/E12933/cluster hc.png)

* DBSCAN
  *  optimal epsilon:0.4 n=6 
  *  Silhouette Score: 0.379
  *  Calinski-Harabasz Score: 7974.005
  *  Davies Bouldin Score: 0.256
      ![DBSCAN optimal ep](C:/Users/eleon/Documents/GitHub/E12933/optimal epsilon dbscan.png)
      ![DBSCAN clusters](C:/Users/eleon/Documents/GitHub/E12933/cluster dbscan.png)

According to our validation metrics, k-means was the best model considering it has the highest Silhouette score and Calinski-Harabasz score. Even if the DBSCAN model gets the best Davies Bouldin score (the lowest the better), we chose k-means because it still has two good validation scores against one. Therefore, we move on to interpreting it's clusters:

  ![Cluster Interpretation (K-means)](C:/Users/eleon/Documents/GitHub/E12933/cluster interpretation (kmeans).png)

From the plotted graph, we can observe the 3 clusters and how they interact with each of the variables in the RFM model. These are the names we chose for each cluster based on the plot:

* The first segment is what we call the 'new customers', as we can see, they have high recency, but still are low on frequency and monetary value, therefore we perceived them as customers who just discovered the brand, so still open to getting to know the brand.

* The second segment we named as 'inactive customers' because they have moderate frequency and monetary value, but their recency is low. That might be the occasion of a customer who bought multiple times before but now they don't buy from the store anymore.

* The third and final is the 'loyal customers' segment. Here, the buyers have high monetary value, above average frequency and moderate recency, so we can consider them our usual costumers, perfect target for email campaigns and other marketing strategies from this department store. 

## Conclusions
Clustering is fundamental for businesses in order to be the most efficient with targeting customers and increase revenues. There are multiple approaches in order to segment customers and there is the need to find the best model. RFM analysis is fundamental to start approaching the problem and it revealed itself really useful to interpret our final results. For our database K-means found itself to be the best fitting model, and the final partition was actually really similar to the initial segmentation made based on the rfm scores, resulting in three final clusters. Between the three clusters, in our opinion, there is one that stands out for frequency and monetary value, and we think that would be the most reccomended to address the new email campaign to them. 

Some unanswered questions were left by the fact that with using as our model k-means we could not fully exploit all the variables in the database that could have given some interesting insights. And for future work the department should develop a strategy in order to get the clients classified as 'new customer' to become part of the 'loyal customer' segment and also to recover the 'inactive customers'.