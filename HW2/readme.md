## Setup
```bash=
bash setup.sh
```

## Process

### Get training data
```bash=
wget http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
unzip Sentiment-Analysis-Dataset.zip
```
### Create pre-train model
```bash=
python3 pre-train.py
```
### Main
```bash=
python3 main.py
```