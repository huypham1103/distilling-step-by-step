# import requests

# # Replace these with your actual values
# ocid = "winp1"
# apikey = "0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM"
# # userid = "your_userid"
# activityid = "A8A195ED-C626-4355-A7BB-76DFE3F24AB3"

# url = "https://api.msn.com/news/Feed"
# params = {
#     "ocid": ocid,
#     "market": "en-us",
#     "query": "Lifestyle",  # Updated to target Lifestyle content
#     "$top": 100,
#     "$skip": 0,
#     "$select": "sourceid,type,url,provider,title,images,publishedDateTime,categories",
#     "apikey": apikey,
#     # "userid": userid,
#     "activityid": activityid
# }

# response = requests.get(url, params=params)

# if response.status_code == 200:
#     data = response.json()
#     # Process the data as needed, e.g., print titles and URLs
#     for article in data['articles']:
#         print(f"Title: {article['title']}, URL: {article['url']}")
# else:
#     print(f"Failed to retrieve data: {response.status_code}, {response.text}")



import requests

url = "https://assets.msn.com/service/community/comments/"

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.5',
    'authority': 'assets.msn.com',
    'origin': 'https://www.msn.com',
    'referer': 'https://www.msn.com/',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sec-gpc': '1',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
}

params = {
    'contentId': 'AA1n8sJI_en-us',
    '$top': 6,
    '$skip': 0,
    '$orderby': 'Usefulness',
    'apikey': '0QfOX3Vn51YCzitbLaRkTTBadtWpgTN8NZLW0C1SEM',
    'activityId': 'A8A195ED-C626-4355-A7BB-76DFE3F24AB3',
    'ocid': 'social-peregrine',
    'cm': 'en-us',
    'it': 'web',
    'user': 'm-30B56E303AA56CB42B6A7DED3BC36D60',
    'wrapodata': 'false'
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    # Process the data as needed
    print(data)
else:
    print(f"Failed to retrieve data: {response.status_code}, {response.text}")
