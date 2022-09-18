import catboost as cb

def get_feature_importances(cb_model):
    return cb_model.get_feature_importance(type=cb.EFstrType.FeatureImportance, prettified=True, thread_count=-1, verbose=False)

