import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    data['duration'] = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
    data.duration = data.duration.apply(lambda td: td.total_seconds() / 60)

    
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    data[categorical] = data[categorical].astype(str)

    train_dicts = data[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X = dv.fit_transform(train_dicts)
    y = data.duration.values

    lr = LinearRegression()
    lr.fit(X, y)
    print(f"Intercept of model : {lr.intercept_}")
    return dv, lr