import os 

with open('demo.txt', 'w') as file:
  size = 10000
  print('demo output size: ', size, file = file)

os.system("git add .")
os.system("git commit -m message")
os.system("git push")