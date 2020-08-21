import requests

# api-endpoint 
URL = " http://127.0.0.1:8000/predict"
  
# params
PARAMS = {'cnae_0':96,
           'cnae_1':2,
           'cnae_2':5,
           'cnae_3':1,
           'score':507.49564, 
           'pout_s12':-13299.04, 
           'pout_c12':590.0,
           'pin_a12':37.622210} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL, params = PARAMS) 

print(r.text)
