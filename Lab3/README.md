## Automatic Panoramic Image Stitching

### Introduction
In this work, we stitch two photos(same objects of different views) together in order to get their panoramic image. In the first, we need to find out two pictures’ interest points and feature 
description. So we can know the same features in two pictures. Because the two photos were taken in different views, we need to calculate the homography matrix by using RANSAC algorithm.  

### Implementation
- please refer HW3 report.pdf to get more details
- enviroment : cv2(version3.4)
- how to run this code
```bash=
python hw3.py
```
### Result
![image](https://user-images.githubusercontent.com/22147510/110294164-77b0f680-802a-11eb-8e1c-27242f4766a6.png)
![image](https://user-images.githubusercontent.com/22147510/110294186-7e3f6e00-802a-11eb-8834-2e7e142c8b75.png)
![image](https://user-images.githubusercontent.com/22147510/110294228-8bf4f380-802a-11eb-80f7-7c13a5a78186.png)
![image](https://user-images.githubusercontent.com/22147510/110294254-931c0180-802a-11eb-97e5-8b68b19e30a5.png)
