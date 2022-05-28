import joblib
import numpy as np
import pandas as pd

run_type = "credit"

l = [35, 50, 60, 70, 85]


def convert(x):
    if run_type == "star":
        return x + 1
    else:
        return l[x]


def predict(model_path, feature, uid_path):
    uid = pd.read_csv(uid_path)[["uid"]]
    model = joblib.load(model_path)
    res = map(convert, model.predict(feature))
    res = pd.DataFrame(data=res, columns=[run_type + "_level"])
    print(uid)
    print(res)
    df = pd.concat([uid, res], axis=1)
    print(df)
    df.to_csv("data/res/" + run_type + "_" + model_path.split("/")[-1].split(".")[0].split("_")[1] + "_res.csv",
              sep=',', index=False, mode='w', line_terminator='\n',
              encoding='utf-8')
    return


if __name__ == '__main__':
    predict_feature = np.load("data/npy/" + run_type + "_predict_feature.npy")
    print(predict_feature)
    predict("data/model_0/" + run_type + "_vote.model", predict_feature, "data/processed_data/" + run_type + "_test.csv")

