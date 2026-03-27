import requests
import json

X = ...
r = requests.post("http:// 0.0.0.0:<PORT>/predict", json=json.loads(X.to_json()))
print(r.json())
