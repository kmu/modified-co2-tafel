import json
from glob import glob
import pandas as pd

data_list = []
for json_path in glob("figures/*.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    sample_id = json_path.split(".")[0].split("/")[-1]
    data_list.append({
        "sample_id": sample_id,
        "map_value": data["map_value"]
    })

df_map = pd.DataFrame(data_list)
df_map.to_csv("map_values.csv", index=False)
