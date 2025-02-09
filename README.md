# UCL-2025-Prediction

This project aims to predict the outcome of the 2025 UEFA Champions League knockout stage using match data from the current season. By collecting and analyzing team performance metrics from the league phase, we built a machine-learning model to evaluate each team’s strength. The dataset includes statistics such as goals, expected goals (xG), points average, and shooting accuracy, ensuring that only up-to-date performance data is used.

We trained a **Random Forest Classifier** to predict match results (Win, Draw, Loss) based on these features. The model was then used to simulate the knockout rounds, forming a bracket where winning teams advanced until a champion was determined.

While the model provides realistic matchups—such as **Real Madrid vs Manchester City**, seen in past seasons—the final results remain uncertain as the tournament is still ongoing. The predictions are based solely on this season's data, without historical comparisons, as team rosters, tactics, and the **new UCL format for 2025** make past seasons less relevant.

The final bracket simulation crowned **Liverpool as Champions**, a plausible outcome given their current form. However, external factors like injuries, tactical changes, and in-game dynamics remain unpredictable.

## Project Structure

The project is organized into multiple Jupyter notebooks and datasets, ensuring a clear workflow from data collection to final predictions.

### Notebooks

- **`DataScraperFootball.ipynb`** – Scrapes match and team statistics from FBref, collecting raw data on Champions League performances.
- **`DataCleaningFootball.ipynb`** – Cleans and processes the scraped data, standardizing formats, assigning team IDs, and handling missing values.
- **`TeamAnalysis.ipynb`** – Analyzes key features for each team, identifying important statistical trends.
- **`Result_Prediction.ipynb`** – Trains the machine learning model and simulates the knockout stage bracket to predict the Champions League winner.

### Datasets

- **`champions_league_stats.csv`** – Raw scraped data, containing team match statistics from the league phase.
- **`ucl_match_data.csv`** – Cleaned and structured data, with irrelevant columns removed and opponent id mappings applied.
- **`processed_dataset.csv`** – Fully processed dataset with engineered features, ready for model training.

### Results

- **`Results/`** – Contains visualizations, model outputs, and the final predicted knockout bracket.

## 1. Data Scraping

The data for the UCL 2025 match predictions was scraped from [FBref - Champions League Stats](https://fbref.com/en/comps/8/Champions-League-Stats), focusing on match statistics and shooting data for teams in the League Phase of the 2024-2025 Champions League. The process was implemented using Python's BeautifulSoup library to parse the HTML content, and requests for sending HTTP requests to the webpages.

The scraping script iterated over a list of team URLs, extracting the following information for each team:

**Match Data**: Using pd.read_html, match statistics were retrieved from the "Scores & Fixtures" table, filtering only the Champions League matches.
**Shooting Data**: Shooting statistics were extracted from the linked "Shooting" pages, including fields like shots (Sh), shots on target (SoT), shot distance (Dist), free kicks (FK), penalties (PK), and penalty attempts (PKatt).
The relevant data was merged for each team, and the team-specific URL was appended for reference.

The columns of the dataset are as follows:

`Date`, `Time`, `Comp`, `Round`, `Day`, `Venue`, `Result`, `GF`, `GA`, `Opponent`, `xG`, `xGA`, `Poss`, `Attendance`, `Captain`, `Formation`, `Opp Formation`, `Referee`, `Match Report`, `Notes`, `Sh`, `SoT`, `Dist`, `FK`, `PK`, `PKatt`, and `Team URL`.

The final scraped data was saved as `champions_league_stats.csv`.

## 2. Data Cleaning

The dataset was cleaned to ensure its readiness for analysis. First, only matches from the "League phase" round were retained, focusing on the group stage of the competition. Irrelevant columns were removed, and key match statistics such as goals scored (GF), goals conceded (GA), shots (Sh), and shots on target (SoT) were preserved.

Teams were assigned unique identifiers (Team IDs) based on their URLs. A new column for `Opponent ID` was created by matching the standardized opponent names with the team URLs.

Missing values were checked and addressed, and the `Result` column was converted into points (3 for a win, 1 for a draw, and 0 for a loss). Additional columns were calculated, including `Goal Difference` (GF - GA), `xG Difference` (xG - xGA), and `Shot Accuracy` (SoT / Sh), providing more insights into team performance.

Finally, the `Date` column was converted into a datetime format to facilitate time-based analysis.

The cleaned data was then saved as `ucl_match_data.csv`, ready for further analysis and modeling.

## 3. Feature Engineering

To ensure the most relevant and up-to-date predictions, we focused only on **this season’s** Champions League data. Unlike traditional models that incorporate historical data, we avoided past seasons since **team compositions, tactics, and the tournament format have changed in 2025**. Using outdated information could misrepresent a team's current strength and performance.

To validate feature importance, we plotted the correlation between key metrics and league phase standings. The graph below highlights how **goal difference, goals scored, and goals conceded align with a team’s ranking in the league phase**:

![Goal Difference Chart](./Results/Feature%20Correlation/Goals%20For:Against.png)

We computed **mean** and **variance** for key performance metrics, such as:

- **Goals Scored (GF)**
- **Goals Conceded (GA)**
- **Goal Difference (GD)**
- **Expected Goals (xG, xGA)**
- **Possession**
- **Shot Accuracy**
- **Set-Piece Performance (FK, PK, PKatt)**

These statistics were then merged into the main dataset to represent both the team’s and its opponent's performance metrics. The final dataset includes both **Team** and **Opponent** statistics for each match.

A correlation analysis was performed between the team and opponent features and the target variable (match result):

![Correlation Bar Chart](./Results/Feature%20Correlation/Feature%20Correlation.png)

Features with low correlation to the result were removed, with a threshold of 0.15 used to select the most relevant features. This step helped in narrowing down the variables to those most strongly related to match outcomes:

![Filtered Correlation Bar Chart](./Results/Feature%20Correlation/Filtered%20Features.png)

An analysis of Expected Goals (Probability of a goal being score from a shot) with respect to match result also shows correlation betweeen xG vs Match Result:

![Expected Goal Correlation Scatter](./Results/Feature%20Correlation/Expected%20Goal%20Difference%20Correlation.png)

The feature-based dataset was then saved as `processed_dataset.csv`, ready for further analysis and model training.

## 4. Model Training

For the model training, a **Random Forest Classifier** was chosen to predict match outcomes based on the features engineered in the previous step. The Random Forest algorithm is an ensemble learning method that constructs multiple decision trees and merges them to improve accuracy and control overfitting.

The training process followed these steps:

1. **Data Splitting**: The dataset was split into a training set and a testing set using an 80-20 split. 80% of the data was used for training the model, and the remaining 20% was held out for testing. This was done using the `train_test_split` function from Scikit-learn, with a random seed set for reproducibility.

2. **Feature Selection**: The features selected based on correlation analysis were used as input to train the model. These features are the most relevant to the match result, such as avg team points, avg goals scored (GF), avg shot accuracy, and avg goal difference.

3. **Model Initialization and Training**: A Random Forest Classifier with 100 estimators (trees) was initialized. The model was trained on the training dataset using the `fit` method. Random forests are known for their ability to handle complex datasets with numerous features without overfitting, thanks to the aggregation of multiple decision trees.

## 5. Model Testing

After training the Random Forest Classifier, the model was tested on the held-out testing set, which had not been used during the training process. The goal was to evaluate how well the model generalizes to unseen data and to check its performance on predicting match outcomes.

The testing process included the following steps:

1. **Prediction**: The trained model was used to predict the match results (Win, Draw, or Loss) on the test set. This was done using the `predict` method of the trained model, which outputs the predicted class (1, 2, or 3 corresponding to Win, Draw, or Loss) for each instance in the test data.

2. **Evaluation**: The model's predictions were compared to the actual outcomes in the test set. The following metrics were used for evaluation:
   - **Accuracy**: Measures the overall correctness of the model's predictions.
   - **Precision, Recall, and F1-score**: These metrics were calculated for each class (Win, Draw, Loss) to assess the model's performance for each match outcome.

   The model's performance was summarized with the following results:
   - Accuracy: **71%**
   - Precision, recall, and F1-score for each match outcome (Win, Draw, Loss) were presented in the classification report.

3. **Classification Report**: The classification report gives more detailed insights into the model's performance, showing precision, recall, and F1-score for each of the three classes (Win, Draw, Loss):
   - **Win (1)**: Precision = 0.61, Recall = 0.90, F1-score = 0.73
   - **Draw (2)**: Precision = 1.00, Recall = 0.12, F1-score = 0.22
   - **Loss (3)**: Precision = 0.81, Recall = 0.72, F1-score = 0.76

4. **Macro and Weighted Averages**:
   - **Macro Average**: Provides an unweighted mean of the precision, recall, and F1-score across all classes. This gives a sense of how well the model performs across all match outcomes, regardless of class imbalance.
   - **Weighted Average**: Averages the precision, recall, and F1-score, but weighted by the number of instances in each class (support). This provides a performance measure that accounts for class imbalances.

Overall, the model demonstrated a good balance between predicting wins and losses, but it faced challenges in predicting draws. This is reflected in the low recall and F1-score for the "Draw" class.

The results of the testing phase provide valuable insights into the strengths and weaknesses of the model, allowing for future improvements and tuning.

## 6. Results

In this phase of the project, we took the predictions generated by the trained Random Forest model and used them to simulate the knockout stages of the Champions League, following the current UCL bracket format.

1. **Bracket Simulation**:
   A bracket was created in Python that matched the winning teams against each other. This bracket followed the current Champions League knockout format, where teams compete in head-to-head rounds, and the winner progresses to the next stage. The predictions made by the model were used to simulate these matchups. The final result showed **Liverpool** as the Champions, which aligns with expectations based on the current season's performance.

2. **Accuracy of Matchups**:
   The model successfully predicted several matchups accurately, such as the well-known matchup between **Real Madrid** and **Manchester City** from last year. However, it's important to note that the predictions are based solely on data from the current season, which includes results up to the present date. As a result, the matchups are accurate up to the point of prediction, but the actual outcomes are yet to be determined.

3. **Limitations**:
   - **No Comparison to Previous Years**: This model is based entirely on the data from the 2025 season. As team rosters change every year and tactical strategies evolve, comparing results to previous years' performances may not be meaningful. Additionally, the new UCL format for 2025 adds another layer of complexity, making comparisons to past seasons even more difficult.

4. **Model's Limitation**: While the model can predict match outcomes based on current data, it cannot account for the dynamic nature of football. Player injuries, team form, and other factors that emerge during the tournament can influence the actual outcome, making it impossible to predict results with complete certainty.

Here is the simulated bracket for the Champions League 2025 based on the model's predictions:

![Champions League 2025 Bracket](./Results/Bracket%20UCL.png)
