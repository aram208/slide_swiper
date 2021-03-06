## Environment
Python 3.6, OpenCV, Selenium

## Quick start

### Step 1
Clone this repo and cd into it

### Step 2
Download the latest chrome driver from: http://chromedriver.chromium.org/downloads and put it anywhere on your file system. Modify the `conf.json` to point to the newly downloaded chrome driver.

### Step 3
Activate your virtual environment, if using one, and install the necessary packages.
```
pip install -r requirements.txt
```

### Step 4
Open up the `conf.json` file and put the url and title for your slides (Google slides only!)

### Step 5
````
python swipe_detect.py
````

## How it works
_...smoothly_
<br>
https://drive.google.com/open?id=1qph97lkD3nvk0jrBQ_h_VYYsSu4cN3Ol

## Tips
The algorithm is using the concept of _running average_. 
When starting, make sure to not have any moving objects in the ROI area of the screen for a second or two (the red boxed area at the top right part which tracks the hand movement).