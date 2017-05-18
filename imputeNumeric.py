def imputeNumeric (dataset, formula):
    """
    Impute numeric values in a dataset usinng linear regression
    dataset = Pandas Dataframe
    formula e.g. 'Y ~ X1 + X2'
    """
    import statsmodels.formula.api as smf
    import pandas as pd
    
    lm = smf.ols(formula = formula, data = dataset)
    res = lm.fit()
    
    temp_train = dataset[pd.isnull(dataset).any(axis=1)].copy()
    temp_train = temp_train.drop(formula.split(None, 1)[0], axis=1).copy()
    
    var_pred = res.predict(temp_train)
    var_pred = var_pred.round(decimals=0)
    
    dataset[formula.split(None, 1)[0]].fillna(var_pred, inplace=True)