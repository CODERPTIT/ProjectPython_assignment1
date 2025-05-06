import requests

DATA = "https://opensource.com/article/22/5/document-source-code-doxygen-linux"
PAGE = requests.get(DATA)

print(PAGE.text)