import requests
import bs4
import re
import sys


url = 'https://www.vocabulary.com/dictionary/' + sys.argv[1]
response = requests.get(url, timeout=5)
soup = bs4.BeautifulSoup(response.text, 'lxml')


print("\n")
print(url.split('/')[-1] + ':')
print('\t' + soup.find('p').text)
print("-"*70)
print("-"*70)

for each in soup.find_all('div', class_="ordinal"):
    for typ in each.find_all('a', class_="anchor"):
        print('(' + typ["title"] + ')')
    for eac in each.findAll('h3', class_="definition"):
#         print(eac.text.split(' '))
#         print(eac.text)
        print(re.sub('\s+',' ',eac.text))
    
    print("\nSynnonyms:")
    for syn in each.findAll('dd'):
        for sy in syn.findAll('a', class_="word"):
            print('\t', sy.text)
        break
#     print(each)
    print("-"*70)
    print("-"*70)
