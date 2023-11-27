# Sales Prediction Based on e-Commerce Patterns Recognition
![eCommerce](Images/eCommerce.jpg)

## Project Overview:
The analysis focuses on predictive analytics techniques applied to an e-commerce setting to predict browsing sessions that result in transactions. Various machine learning methodologies are extensively reviewed and Random Forest is ultimately selected due to its consistent high performance in similar studies as well.

The methodology outlines steps for data preprocessing, handling imbalanced data, feature selection, and model parameter tuning. Results are thoroughly discussed, including evaluation metrics, a confusion matrix, R.O.C. curve analysis, and identification of influential features for online purchases.

Furthermore, the analysis delves into the practical implications of predictive analytics in business, highlighting potential applications in marketing strategies, dynamic pricing, platform optimisation, inventory management, and strategic planning.

In summary, the analysis offers a comprehensive examination of predictive modelling in eCommerce, presenting a well-structured methodology and discussing both technical and practical implications, while also suggesting areas for further improvement and development.

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
| BounceRates              | Percentage of visitors entering and leaving after  |
|                          | visiting only one page                             |
| ExitRates                | Percentage of visitors leaving the website from a  |
|                          | given page                                         |
| PageValues               | Average value of pages visited before completing a |
|                          | transaction                                        |
| SpecialDay               | Closeness to a special day                         |
| Month                    | Month of the year                                  |
| OperatingSystems         | Machine's operating system                         |
| Browser                  | Browser used during the session                    |
| Region                   | Where the session originated from                  |
| TrafficType              | Channel the session originated from                |                                 
| VisitorType              | Returning vs new visitor                           |
| Weekend                  | Weekend True vs False                              |
| Revenue                  | Purchase True vs False (target label)              |