import os 

with open('demo.txt', 'w') as file:
  print('demo output', file = file)

os.system("git add .")
os.system("git commit -m message")
os.system("git push")