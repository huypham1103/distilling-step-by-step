import requests as r
import json
import os

# API key
api_key = '416980c8abafe6aa8c7740f64d6318'

url =  "https://thuvienphapluat.vn/van-ban/Tai-nguyen-Moi-truong/Quyet-dinh-1881-QD-TTg-2021-Danh-sach-co-so-su-dung-nang-luong-trong-diem-2020-493862.aspx?v=d"

headers = {
    "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.6",
    "sec-ch-ua": "\"Brave\";v=\"117\", \"Not;A=Brand\";v=\"8\", \"Chromium\";v=\"117\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "image",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1"
  }


response = r.get(url, headers=headers)

print(response)

print(response.text)

