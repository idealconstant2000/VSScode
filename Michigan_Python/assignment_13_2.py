import urllib.request, urllib.parse
import json, ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = input('Enter location: ')
if len(url) < 1 : 
    url = 'http://py4e-data.dr-chuck.net/comments_42.json'

print('Retrieving', url)

uh = urllib.request.urlopen(url, context=ctx)
data = uh.read().decode()
print('Retrieved',len(data),'characters')

info = json.loads(data)

nums =[]


for item in info["comments"]:
    nums.append(int(item["count"]))
#    print(result.text)

print('Count:', len(nums))
print('Sum:', sum(nums))