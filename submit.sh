# manual submit of the challenge
docker build -t protein-language-docker .

# reinstall biolib incase its updated
pip3 install pybiolib

# push submission
biolib push $1 --path submission
