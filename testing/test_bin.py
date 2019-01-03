import urllib
from inscriptis import get_text


def test_bin(fitted_model)
  url = "https://sentence.yourdictionary.com/hate"
  html = urllib.request.urlopen(url).read().decode('utf-8')
  text = get_text(html)
  text=text.strip()
  text=text.split('.')

  model=
  all_sen=""
  for sen in text:
  #     x = clf.predict([sen])
      x=count_vect.transform([sen])

      x=a.predict(x)
      if x==True:
          all_sen=all_sen+sen+' 1'
      else:
      
  print(all_sen)
