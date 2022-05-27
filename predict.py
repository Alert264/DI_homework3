import joblib
import numpy as np


def predict(model_path, feature):
    model = joblib.load(model_path)
    res = model.predict(feature)
    print(res)


if __name__ == '__main__':
    predict_feature = np.load("data/npy/star_predict_feature.npy")
    print(predict_feature)
    predict("data/model/star_LR.model", predict_feature)
