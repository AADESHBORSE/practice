import requests
from bs4 import BeautifulSoup

# 🔴 Replace URL
url = "YOUR_URL"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# 🔴 Replace tag/class
data = soup.find_all("TAG_NAME")

for item in data:
    print(item.text)
