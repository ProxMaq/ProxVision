# Yolov4-Detector-and-Distance-Estimator

*Find distance from object to camera using YoloV4 object detector, here we will be using single camera :camera:, detailed explanation of distance estimation is available another repository* [**Face detection and Distance Estimation using single camera**]

implementation detail available on [_**Darknet**_](https://github.com/pjreddie/darknet)


---

## TO DO

- [x] Finding distance of mutiple object at same time.

## Installation you need opencv-contrib-python

[opencv contrib](https://pypi.org/project/opencv-contrib-python/)

### **windows**

```pip
pip install opencv-contrib-python==4.5.3.56
```

### **Linux or Mac**

```pip
pip3 install opencv-contrib-python==4.5.3.56
```

then just clone this repository and you are good to go.

I have use tiny weights, check out more on darknet github for more

---

## Add more Classes(Objects) for Distance Estimation

You will make changes on these particular lines [***DistanceEstimation.py***](https://github.com/Asadullah-Dal17/Yolov4-Detector-and-Distance-Estimator/blob/master/DistanceEstimation.py#L50-L56)
```python
if classid ==0: # person class id 
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
elif classid ==67: # cell phone
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
    
# adding more classes for distnaces estimation 

elif classid ==2: # car
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])

elif classid ==15: # cat
    data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
# in that way you can include as many classes you want 

    # returning list containing the object data. 
return data_list

```

## Reading images and getting focal length

You have to make changes on these lines 📝 [***DistanceEstimation.py***](https://github.com/Asadullah-Dal17/Yolov4-Detector-and-Distance-Estimator/blob/master/DistanceEstimation.py#L69-L76)
there two situations, if the object(classes) in single image then, here you can see the my reference image <img src='ReferenceImages/image4.png' width=200> 
it has to two object, *person* and *cell phone*
```python
# reading refrence images 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')
# calling the object detector function to get the width or height of object
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

# getting pixel width for cat
cat_data = object_detector(ref_person)
cat_width_in_rf = person_data[2][1]

# getting pixel width for car
car_data = object_detector(ref_person)
car_width_in_rf = person_data[3][1]

```
if there is single class(object) in reference image then you approach it that way 👍
```python
# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/person_ref_img.png')
ref_car = cv.imread('ReferenceImages/car_ref_img.png.png')
ref_cat = cv.imread('ReferenceImages/cat_ref_img.png')
ref_mobile = cv.imread('ReferenceImages/ref_cell_phone.png')

# checking object detection on reference image 
# getting pixel width for person
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

# getting pixel width for cell phone
mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

# getting pixel width for cat
cat_data = object_detector(ref_cat)
cat_width_in_rf = person_data[0][1]

# getting pixel width for car
car_data = object_detector(ref_car)
car_width_in_rf = person_data[0][1]
# then you find Focal length for each

```


