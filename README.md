### Table of Contents

1. [Installation](#installation)
2. [Run Predcition](#prediction)
3. [Train line recognition model](#trainOCR)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`. 

To download and setup the necessary folders, run

``./setup.sh``

## Run Prediction <a name="prediction"></a>

To test the model for line rocognition on IAM dataset, run

```python recognition.py -i [path to line image directory]```

The code will write the predicted text into a new *.txt* file with the same file name as the input line image in a new ``./results`` directory 

By default, the code generates prediction from the best OCR model along with the brute force post-edit method. This combination performs best compare ot other methods. To try other post-edit methods,run:


```python recognition.py -i [path to line image directory] -p [provide post edit option]```

Four post edit options are available:
  - ``brute`` (replace the misspelled word with a word that needs minimum number of edits)
  - ``candidate`` (use language model to find the best replacement of the misspelled word)
  - ``neuspell`` (a neural spell correction tool that uses a transformer based language model to build a spell correction model)
  - ``neuspell-edit`` (only replace the misspelled word with neuspell model prediction and keep others unchanged)
  
 Example: ```python recognition.py -i [path to line image directory] -p brute```
 
 
 ## Train Line recognition model <a name="trainOCR"></a>
 
