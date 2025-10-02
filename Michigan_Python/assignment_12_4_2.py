import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import ssl # defauts to certicate verification and most secure protocol (now TLS)

# Ignore SSL/TLS certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Inputs
url = input("Enter URL: ").strip()
count = int(input("Enter count: ").strip())
position = int(input("Enter position: ").strip())  # 1-based

def extract_name_from_url(u):
    m = re.search(r'known_by_([A-Za-z]+)\.html', u)
    return m.group(1) if m else None

# Retrieve initial page + each hop
for i in range(count + 1):  # include the initial page
    print("Retrieving:", url)
    if i == count:
        break  # we've reached the final page to report
    # Fetch and parse
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, "html.parser")
    # Get all links and pick the one at the given position
    links = soup.find_all("a")
    url = links[position - 1].get("href")
