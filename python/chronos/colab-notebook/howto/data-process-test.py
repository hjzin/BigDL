from bigdl.chronos.data import TSDataset
import pickle
import pandas as pd

# load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# load the data to be predicted
df = pd.read_csv("inference_data.csv", parse_dates=["timestamp"])


def preprocess_during_deployment(df, scaler):
    tsdata = TSDataset.from_pandas(df,
                                   dt_col="timestamp",
                                   target_col="value",
                                   repair=False,
                                   deploy_mode=True)
    tsdata.gen_dt_feature(features=["WEEKDAY", "HOUR", "MINUTES"])\
          .scale(scaler, fit=False)\
          .roll(lookback=48, horizon=24, is_predict=True)
    data = tsdata.to_numpy()
    return tsdata, data

if __name__ == "__main__":
    from bigdl.chronos.metric.forecast_metrics import Evaluator

    latency = Evaluator.get_latency(preprocess_during_deployment, df, scaler)
    print("process time", latency)