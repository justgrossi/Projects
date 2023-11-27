# Machine Learning: a Comparative Analysis
![eCommerce](Images/Comparative.jpg)

## Project Overview:
The predictive ability of five machine learning algorithms is compared across three diverse datasets. The analysis addresses real-world challenges in different industries: churn prediction in subscription-based businesses, term deposit subscriptions in relation to telemarketing campaigns, and energy consumption forecasting.

A structured approach is adopted to develop data understanding, preparation, algorithm selection, modelling, and evaluation. All the steps for data preprocessing, handling imbalanced data, feature selection, engineering, and model parameter tuning are presented.

Models are extensively evaluated using accuracy, precision, sensitivity, specificity, R.O.C. curves, A.U.C., M.S.E., R.M.S.E., R-squared, and confusion matrices.

Furthermore, the analysis delves into the practical implications of predictive analytics in business discussing both technical and practical implications, while also suggesting areas for further improvement and development.

## Data Dictionary:

| Feature                  | Description                                        |
|--------------------------|----------------------------------------------------|
| Administrative           | Number of times a page showing administrative      |
|                          | information was visited                            |
| AdministrativeDuration   | Seconds spent on an administrative page            |
| Informational            | Number of times an informational page was visited  |
| InformationalDuration    | Seconds spent on an informational page             |
| ProductRelated           | Number of times a product related page was visited |
| ProductRelatedDuration   | Seconds spent on a product related page            |
| BounceRates              | Percentage of visitors leaving after visiting only |
|                          | one page                                           |
| ExitRates                | Percentage of visitors leaving the website from a  |
|                          | given page                                         |
| PageValues               | Average value of pages visited before completing a |
|                          | transaction                                        |
| SpecialDay               | Closeness to a special day                         |
| Month                    | Month of the year                                  |
| OperatingSystems         | Machine operating system.                          |
| Browser                  | Browser used during the session                    |
| Region                   | Location the session originated from               |
| TrafficType              | Channel the session originated from                |                                 
| VisitorType              | Returning vs new visitor                           |
| Weekend                  | Weekend True vs False                              |
| Revenue                  | Purchase True vs False (i.e. target label)         |