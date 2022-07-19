from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def execute_gbr(feature_df, train_df, test_df):
    rng = np.random.default_rng()  # necessary to reset the random seed within each parallel process
    rand_seed = rng.integers(0, 100)
    regressor = GradientBoostingRegressor(n_estimators=20000, max_depth=8,
                                          min_samples_split=6, learning_rate=0.005,
                                          loss='ls', subsample=0.7,  # 'ls' deprecated but used for HPCC compatibility
                                          max_features='sqrt', random_state=rand_seed)

    train_features = feature_df[feature_df.index.isin(train_df.id)].to_numpy()
    test_features = feature_df[feature_df.index.isin(test_df.id)].to_numpy()
    train_labels = train_df.num.to_numpy()

    regressor.fit(train_features, train_labels)
    test_predicted = regressor.predict(test_features)

    return test_predicted


def train_test_correlation(correlation_df, feature_df, train_df, test_df):
    for i in range(len(correlation_df)):
        test_predicted = execute_gbr(feature_df=feature_df, train_df=train_df, test_df=test_df)
        # plt.scatter(test_predicted,test_df["num"].to_numpy())
        correlation_df.iloc[i, 0] = np.corrcoef(test_predicted, test_df["num"].to_numpy())[1, 0]
        correlation_df.iloc[i, 1] = mean_squared_error(test_df["num"].to_numpy(), test_predicted, squared=False)
    # plt.show()
    return correlation_df
