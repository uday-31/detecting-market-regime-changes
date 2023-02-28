# detecting-market-regime-changes

Used hidden Markov models to label regimes on S&P 500 data, then trained three classifiers (naive Bayes, logistic regression, and support vector machines) to predict regimes with the intent of improving an existing trading strategy.

This project was done for the requirements of the class 46-927 Machine Learning II. The group members are Dhruv Baid, Rohan Prasad, and Sarthak Vishnoi, and Uday Sharma.

# Abstract

This project used machine learning to detect and predict changes in the market regime with the aim of improving existing trading strategies. Our approach used directional change (DC) indicators derived from the price series of the S\&P 500 index. Our analysis had two parts: using a hidden Markov model on the training data to detect regimes retrospectively, and training three classifiers -- naive Bayes (NBC), logistic regression (LR), and support vector machines (SVM) -- on the labeled data. We evaluated the models based on the performance of a trading strategy that used the regime data and compared it to baseline strategies that didn't take regimes into account. We were able to improve the baseline trading strategies for all three models. The NBC performed the best.

# Research Questions

- Can these regimes help us improve existing trading strategies?
- Does any particular model perform better than the others in the context of the trading strategy?
- What characteristics do these regimes show?
