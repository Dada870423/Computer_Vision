## Structure From Motion

### Introduction
The goal of this homework is using two pictures to reconstruct 3D model. In homework 3, we had learned the method of finding the same points in two pictures taken in different angles. In order to find 
fundamental matrix, we apply 8-points algorithm on previous points we found and ratio test. After getting fundamental matrix, we can calculate 4 possible essential matrices. The final step is examining 4 
possible directions of camera, and then picking the best result with maximum number of points in front of camera. 

## Implementation
- please refer HW4 report.pdf to get more details
- how to run this code
```bash=
python SfM.py case
case : 1=Mesona, 2=Statue, 3=nctu
```

## Result
![image](https://user-images.githubusercontent.com/22147510/110297298-49351a80-802e-11eb-8c1a-c18aa29f5b75.png)

![image](https://user-images.githubusercontent.com/22147510/110297354-58b46380-802e-11eb-8b03-b4b406afbb62.png)
