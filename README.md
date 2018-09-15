## Environment
Python 3.6, OpenCV, Selenium

## Quick start

### Step 1
Clone this repo and cd into it

### Step 2
Download the latest chrome driver from: http://chromedriver.chromium.org/downloads and put it anywhere on your file system. I usually put it under the _utils_ folder.

Modify the conf.json to point to the newly downloaded chrome driver.

### Step 3
Install the necessary packages
```
pip install -r requirements.txt
```

### Step 4
Open up the conf.json file and put the url and title for your slides (Google slides only!)

#### Step 5
````
python swipe_detect.py
````

## How it works
_...smoothly_