import requests
r = requests.get("http://172.28.0.2/")
print(r.status_code)
print(r.encoding)
print(r.apparent_encoding)
print(r.text)