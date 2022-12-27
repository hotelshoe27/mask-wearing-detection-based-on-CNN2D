# mask-wearing-detection-based-on-CNN2D  

**This CNN based model detects whether a mask is worn using OpenCV, cvlib.**

See requirements.txt before testing and check detail through source code👀

and Plz follwoing this below💨

---

### 1_Create a dataset

- Create a dataset using 'Get_img_data.py' like this below.(I'm not so handsome, so I blurred it😅)  
- Automatically capture the face area at regular intervals. At this time, it's better to capture the left and right sides together.  

![data_0](https://user-images.githubusercontent.com/120359150/209596313-93d9177c-6c9c-416e-bfc7-c788eb8ca2c4.PNG) ![data_1](https://user-images.githubusercontent.com/120359150/209596350-b03cf329-6ba4-4dbf-bf39-32b9bf29e6a1.PNG)

---

### 2_Model training

- Train a model based on CNN2D through 'Model_build.py' file. In this project case, it used three CNN2D and See source code for details about model.  
- Also, The sample test results were got from built model through each 150 mask and nomask images.  

---

### 3_Test result (images)
![result_0](https://user-images.githubusercontent.com/120359150/209530456-5eacb5a8-5c56-4c4e-bf97-d7de9d9d2c39.PNG)  
![result_1](https://user-images.githubusercontent.com/120359150/209530484-d67aafbf-4470-451c-97b1-f4e43d48bcd6.PNG)

---

### 4_Test result (Video)
Plz testing yourself😅
