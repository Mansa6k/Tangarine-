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

![Baseline_Strategy_returns.jpg](https://github.com/Mansa6k/Tangarine/blob/main/Images/Baseline_Strategy_returns.jpg)

---
