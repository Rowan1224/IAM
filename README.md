### Table of Contents

1. [Installation](#installation)
2. [Run Predcition](#prediction)
3. [Train line recognition model](#trainOCR)
4. [Evaluate Line recognition model and post-edit methods](#eval)
5. [Train neuspell model](#neu)

## Installation <a name="installation"></a>

Requires `python 3.6+`

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
 
 To train the Line recognition model, run:
 
 ```
    python dataset.py -i [raw input image directory] -o [desired output image directory] -l [file location for the label file]
    cd line
    python train -d [dataset directory path (output image directory)] -e [epochs] 
 ```
  
   ## Evaluate Line recognition model and post-edit methods <a name="eval"></a>
  
  To evaluate prediction performance after applying post-edit methods on the Line recognition model outputs, run:
  
  ```
      python eval.py -i [prediction file*] -p [post-edit method]
      
  ```
  
  ## Train neuspell model  <a name="neu"></a>
 
  ```
    cd post/neuspell
    python create_dataset.py -i [input directory that contains prediction files] -o [desired output directory] -e [the line recognition model epoch]
    python train -e [epochs] 
    
 ```
