# THIS FILE IS ONLY FOR MANUAL SUBMISSION

docker build -t cbh2021:latest .

# reinstall biolib incase its updated
pip3 install pybiolib

# push submission
biolib push $1 --path submission
