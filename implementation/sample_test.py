import os 
import git

# Optionally, you can also set the branch to track the remote branch
# repo.heads[branch_name].set_tracking_branch(repo.remotes.origin[branch_name])

# Now the changes are pushed to the remote repository

with open('demo.txt', 'w') as file:
  size = 10000
  print('demo output size: ', size, file = file)

# Define the path to your Git repository
repo_path = './negation_for_common_sense'

#https://github.com/hmfattah/negation_for_common_sense

# Initialize a Git repository object
repo = git.Repo(repo_path)

# Define the branch you want to push to (e.g., 'main' or 'master')
branch_name = 'main'

# Fetch the remote repository to ensure you have the latest changes
repo.remotes.origin.fetch()

# Switch to the branch you want to push
repo.git.checkout(branch_name)

# Make some changes to your files (e.g., modifying a file)

# Add the changes to the staging area
repo.index.add(['demo.txt'])

# Commit the changes
commit_message = "Your commit message here"
repo.index.commit(commit_message)

# Push the changes to the remote repository
repo.remotes.origin.push(branch_name)