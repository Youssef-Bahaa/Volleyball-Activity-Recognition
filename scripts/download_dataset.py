from kaggle.api.kaggle_api_extended import KaggleApi
import os

os.makedirs("data", exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Downloading dataset...")

api.dataset_download_files(
    "sherif31/group-activity-recognition-volleyball",
    path="data/",
    unzip=True
)

print("Done!")