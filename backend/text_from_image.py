import requests

api_key = 'K86224171188957'  # clé gratuite
url = 'https://api.ocr.space/parse/image'

with open('temp_page_2.png', 'rb') as f:
    r = requests.post(url,
                      files={'image': f},
                      data={'apikey': api_key, 'language': 'fre'})

result = r.json()

if 'ParsedResults' in result and result['ParsedResults']:
    print(result['ParsedResults'][0]['ParsedText'])
else:
    print("OCR failed or no text found")
    if 'ErrorMessage' in result:
        print("Error:", result['ErrorMessage'])
    else:
        print(result)



