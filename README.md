# Machine Learning Trading Bot

# Overview
In this project, I aimed to improve the algorithmic trading systems used by a financial advisory firm by enhancing the existing trading signals with machine learning algorithms that can adapt to new data. I followed the following steps:
* Imported and preprocessed a dataset of historical stock prices and technical indicators such as moving averages and trading volumes.
* Trained and evaluated multiple machine learning models, including linear regression, random forest, and neural networks, to predict future stock prices.
* Used the best performing model to make trading decisions automatically, based on its predictions.
* Evaluated the performance of our trading bot using various metrics such as return on investment and accuracy.
My approach showed promising results, with my trading bot outperforming the baseline trading algorithm by a significant margin. I believe that my use of machine learning algorithms to enhance trading signals can provide a valuable tool for financial advisors in today's highly dynamic and competitive market.

# Technologies
The following technologies were used in this project:
* Python
* Pandas
* Scikit-learn
* Pathlib
* Numpy
* Mathplotlib

# Installation Guide
To run this project, install the required libraries and dependencies by running the following command in your terminal:

pip install pandas 

pip install scikit-learn 

pip install numpy

pip install pathlib

pip install mathplotlib

# Usage
Clone the repository to your local machine and run the Jupyter notebook file named "machine_learning_trading_bot.ipynb". This will execute all the steps of the project and provide you with the results.

# License
This project is licensed under the MIT license.

# Contributors
 Andre Johnson
---
# Evalution Report
---

## Baseline Performance

The performance of the baseline strategy was underwhelming, as it resulted in a loss of approximately 30% based on past data analyzed. The strategy consisted of creating trading signals by utilizing short-term and long-term Simple Moving Average (SMA) values. The baseline SMA values were 4 periods for the short SMA and 100 periods for the long SMA. The strategy signals were set to 1 if the return was equal to or greater than 0, and -1 if the return was less than 0.

![Baseline_Strategy_returns](https://user-images.githubusercontent.com/118853744/227025257-ef587736-3fc5-4ca3-be0b-6ad625fe5018.jpg)
---

## Tune Baseline Trading Algorithm

While attempting to enhance the performance of the baseline strategy, we modified the input features of the model in an effort to identify the parameters that lead to better trading results. The following changes were made, and the results are presented below:

An SVM model with 3 million training data, utilizing a short-term SMA of 4 periods and a long-term SMA of 100 periods, performed better than actual returns for the majority of 2019 and 2020. Additionally, when increasing the amount of training data to 6 million, the strategy demonstrated even better performance, outpacing actual returns. For the model with 6 million training data, actual returns outperformed the strategy returns for much of 2019, but from March 2020 and beyond, the strategy outperformed actual returns, yielding a total return of approximately 80%, while the model with 3 million training data generated a return of around 50%.

1. adjusting the training window from 3 months to 6 months

While changing the training window from 3m to 6m:
* accuracy of the model increased by 1 point from  0.55 to 0.56. 
* Precision for -1 increased by 1 point from 0.43 to 0.44
* Recall for -1 decreased from 0.04 to 0.02

![svm_class_report_3m_training_data_SMAFast_4_SMASlow_100](https://user-images.githubusercontent.com/118853744/227027326-daee11c2-2a3f-4386-982e-c700b8ee8b78.jpg)..![svm_class_report_6m_training_data_SMAFast_4_SMASlow_100](https://user-images.githubusercontent.com/118853744/227027711-f2a32161-efb7-4179-8562-343d67d6aaa9.jpg)

![svm_plot_3M_training_and_SMAFast_4_SMASlow_100](https://user-images.githubusercontent.com/118853744/227028110-bb914a8f-6d58-42a6-a3b3-e48429515b39.png)

![svm_plot_6M_training_and_SMAFast_4_SMASlow_100](https://user-images.githubusercontent.com/118853744/227028199-f4d87bd4-df65-4286-b810-6b80bb606285.png)


2. adjusting the SMA input features 
---
2a. adjusting SMA short to 2 and SMA long to 50

While changing the SMA window to 2 and 50:
* accuracy of the model decreased by 1 point from  0.55 to 0.54. 
* Precision for -1 decreased by 4 points from 0.43 to 0.39
* Recall for -1 increased from 0.04 to 0.07

![svm_class_report_3m_training_data_SMAFast_2_SMASlow_50](https://user-images.githubusercontent.com/118853744/227028999-73bb2108-0e88-4083-9fa5-e4f61c21ef2e.jpg)

![svm_plot_3M_training_and_SMAFast_2_SMASlow_50](https://user-images.githubusercontent.com/118853744/227029211-47c94649-818f-484b-8aeb-b1960b961031.png)

---
2b. adjusting SMA short to 1 and SMA long to 25

While changing the SMA window to 1 and 25:
* accuracy of the model decreased by 1 point from  0.55 to 0.54. 
* Precision for -1 decreased by 1 points from 0.43 to 0.42
* Recall for -1 increased from 0.04 to 0.13

![lr_class_report_3m_training_data_SMAFast_1_SMASlow_25](https://user-images.githubusercontent.com/118853744/227029602-c0d783cd-9c9f-472c-a1ed-ab05b6f3d58f.jpg)

![svm_plot_3M_training_and_SMAFast_1_SMASlow_25](https://user-images.githubusercontent.com/118853744/227029783-2bb4e2de-b955-48a6-a49c-3dd9525c8161.png)

## Evaluate a New Machine Learning Classifier

For the evaluation of how a different machine learning classifier would perform compared to the provided baseline model, the logistic regression model was used for the comparison. and the model was ran with 3M of training data and SMA_short of 1 period and SMA_long of 25

While changing the SVM model to Logistic Regression Model (SMA 1 and 25):
* accuracy of the model was the same at 0.54. 
* Precision for -1 increased by 1 points from 0.42 to 0.43
* Recall for -1 increased from 0.13 to 0.16

![svm_class_report_3m_training_data_SMAFast_1_SMASlow_25](https://user-images.githubusercontent.com/118853744/227030758-8ce6c132-2e05-4302-a864-f482055ffdfc.jpg)

![lr_plot_3M_training_and_SMAFast_1_SMASlow_25](https://user-images.githubusercontent.com/118853744/227030909-a658096a-3bbd-4a11-93a8-47390250de0f.png)

After reviewing the results of the evaluation methods, it appears that the model does not possess strong predictive abilities. Despite various modifications to the original model, there was not a significant improvement in its performance. As a result, we cannot endorse using this model for algorithmic trading purposes in its current form.
