import json
import urllib.request, ssl

# Ignore SSL errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def explore_json(obj, indent=0):
    """Recursively print keys and values in a JSON-like dict/list"""
    space = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{space}{k}:")
            explore_json(v, indent + 1)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            print(f"{space}[{i}]:")
            explore_json(item, indent + 1)
    else:
        print(f"{space}{obj}")
        
url = "https://py4e-data.dr-chuck.net/opengeo?q=South+Federal+University"
data = urllib.request.urlopen(url, context=ctx).read().decode()
js = json.loads(data)

# Explore the structure
explore_json(js)