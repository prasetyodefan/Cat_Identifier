# Cat Breed Identifier With SVM
This project is a backup of the code and data used in my thesis on "Cat Identifier with Support Vector Machines (SVM)". In my thesis, I explore the use of SVMs for classifying images of cats breed. The code and data in this repository are the implementation and results of my experiments.


## Project Structure

The repository contains the following files and directories:
- [` Asset `](https://github.com/prasetyodefan/Cat_Identifier/tree/main/asset)  : A directory containing the dataset of cat breed images used in the experiments.
- [` Code  ` ](https://github.com/prasetyodefan/Cat_Identifier/tree/main/code)  : A directory containing the Python code used for training the SVM and testing its performance.

## Usage 

Steps to use the code in this repository
1. Install [Python Version 3.10.0](https://www.python.org/downloads/release/python-3100/)
2. [Clone the repository](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop) to your local machine. 
3. Install [Required Library](https://github.com/prasetyodefan/Cat_Identifier/blob/main/requirements.txt) with this code in cmd/terminal <br>
```python   

pip install -r requirements.txt    

```

## Example Code

- Cat Face Detection
- Crop And Resize Function
- [Otsu Thresholding](https://github.com/prasetyodefan/Rand-code-backup/blob/main/Otsu.md)
- [PHOG Descriptor](https://github.com/prasetyodefan/Rand-code-backup/blob/main/Phog.md)
- [SVM Classifier](https://github.com/prasetyodefan/Rand-code-backup/blob/main/SVM.md)


## Flowchart 

<p align="center">
  
<img src="https://user-images.githubusercontent.com/20703698/223105125-708afb87-c83a-43e3-8fe0-06627e9d5b3e.png">

</p>

SVM is not suitable for large datasets because of its high training time and it also takes more time in training compared to Naïve Bayes. It works poorly with overlapping classes and is also sensitive to the type of kernel used.


