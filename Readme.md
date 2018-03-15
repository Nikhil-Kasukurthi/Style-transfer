

# Style Transfer
To increase the vistor interaction in the museum, we propose the Style transfer module. 

When the user likes a particular painting, he/she can scan the name tag and try out their own images in the style of the painting. 

Here is an example. 

|Original Image | Style Image| Transfered Image
|---|---|---|
|![Shenyang](https://raw.githubusercontent.com/Nikhil-Kasukurthi/Style-transfer/master/images/content/shenyang3.jpg)|![Vasudeva carrying Krishna](https://github.com/Nikhil-Kasukurthi/Style-transfer/raw/master/images/museum_styles/66-14.JPG)|![Transfered Image](https://github.com/Nikhil-Kasukurthi/Style-transfer/raw/master/static/shenyang3.jpg)|


## Install Instructions

To run the server

```python server.py```

If you want to test the model run

```python main.py eval --content-image images/content/shenyang.jpg --style-image images/museum_styles/43-15.jpg --model /21styles.model --content-size 1024```


## API Documentation

### Base URL
```
api.team-iris.me/style-transfer
```
#### Datset of styles available

```/dataset```
 
 ```
 METHOD: GET
 Parameters: None
 
 Response: 
           {
              "images": [
                  {
                      "Title": "The Sword of Damocles",
                      "Database ID": "22-4544",
                      "Link": "https://drive.google.com/open?id=1oVrDaOCQAslJMTyWP-9LW0COj5ZG1phD"
                  }, ... (clipped)
               ]
          }
 ```
 
 ```/upload```
 
 ```
 METHOD: GET
 Parameters: None
 
 Response:
           {
              "style_image": "/static/flowers.jpg"
            }
 
 ```
