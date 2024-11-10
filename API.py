import requests
import json

URL = 'https://cloud-api.yandex.net/v1/disk/resources'
TOKEN = 'y0_AgAAAAAusKjvAAyy_QAAAAEWkhirAACI4Vt33mRE9Zgca33bEVNQa_UNTg'
headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': f'OAuth {TOKEN}'}

print(requests.get("https://cloud-api.yandex.net/v1/disk/resources/files", headers=headers).json())
