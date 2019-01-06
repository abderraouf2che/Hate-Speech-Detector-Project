import urllib
from inscriptis import get_text

#This function tests the fitted model and detects if the sentences from the webpage are of the class or not,it returns them 
def test_bin(fitted_model,link_web):
  url = link_web
  html = urllib.request.urlopen(url).read().decode('utf-8')
  text = get_text(html)
  text=text.strip()
  text=text.split('.')

  model=fitted_model
  all_sen=""
  for sen in text:
  #     x = clf.predict([sen])
      x=count_vect.transform([sen])

      x=model.predict(x)
      if x==True:
        all_sen=all_sen+sen+' 1'
  return all_sen

  
#This function tests the fitted multi model and detects if the sentences 
#from the webpage belong to one of the classes or not,it returns them 
  
def testing_multi(fitted_model, link_web):
  url =link_web
  html = urllib.request.urlopen(url).read().decode('utf-8')
  text = get_text(html)
  text=text.strip()
  text=text.split('.')

  model=fitted_model
  all_sen=""
  for sen in text:
  #     x = clf.predict([sen])
      x=count_vect.transform([sen])

      x=model.predict(x)
      if x==1:
        all_sen=all_sen+sen+' 1'
      elif x==2:
        all_sen=all_sen+sen+' 2'
      elif x==3:
        all_sen=all_sen+sen+' 3'
  return all_sen
  
