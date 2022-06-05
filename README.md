### Table of Contents

1. [Installation](#installation)
2. [Run Predcition](#prediction)
3. [Train line recognition model](#trainOCR)
4. [Evaluate Line recognition model and post-edit methods](#eval)
5. [Train neuspell model](#neu)

## Installation <a name="installation"></a>

Requires `python 3.6+`

Clone this repository: `git clone  https://github.com/Rowan1697/IAM.git`

To install all the libraries: `pip install -r requirements.txt`. 

To download and setup the necessary folders: ``./setup.sh``


## Run Prediction <a name="prediction"></a>

To test the model for line rocognition on IAM dataset, run

```python3 recognition.py -i [path to line image directory]```

The code will write the predicted text into a new *.txt* file with the same file name as the input line image in a new ``./results`` directory 

By default, the code generates prediction from the best OCR model along with the brute force post-edit method. This combination performs best compare ot other methods. To try other post-edit methods,run:


```python3 recognition.py -i [path to line image directory] -p [provide post edit option]```

Four post edit options are available:
  - ``brute`` (replace the misspelled word with a word that needs minimum number of edits)
  - ``candidate`` (use language model to find the best replacement of the misspelled word)
  - ``neuspell`` (a neural spell correction tool that uses a transformer based language model to build a spell correction model)
  - ``neuspell-edit`` (only replace the misspelled word with neuspell model prediction and keep others unchanged)
  - ``no`` (no post edits to perform)
  
 Example: ```python3 recognition.py -i [path to line image directory] -p brute```
 
 
 ## Train Line recognition model <a name="trainOCR"></a>
 
 To train the Line recognition model, follow this steps:
 
 ### Data Pre-Processing
 
  Download the IAM dataset. It is expected that the archive contains an folder with all line images and a txt file containing ground truths.
  Run the dataset.py file from the main project folder.Try ``` python3 dataset.py -h ``` to know more about arguments
  ```
  python3 dataset.py 
  -i [raw input image directory] 
  -o [desired output image directory] 
  -l [file location for the label file]
  
  ```
  This code should create three folders: train, test, and valid containing line images. A pickle will also be created to store the ground truth transcriptions. 
  
  ### Train Model
  
  To train the model run the ``` train.py ``` file from ``line`` directory
 
 ```
    cd line
    python3 train.py 
    -d [dataset directory path (required)] #directory where pre-processed image and labels are present
    -e [epochs (defualt 10)] #desired number of epochs
    -m [model directory (optional)] #directory path to save models
    -o [output directory (optional)] #directory path to save outputs and results
    -c [boolean] #use this argument to continue traininge from last checkpoints. By default it will start from begining
 ```
  
   ## Evaluate Line recognition model and post-edit methods <a name="eval"></a>
  
  To evaluate prediction performance after applying post-edit methods on the Line recognition model outputs, run:
  
  ```
      python3 eval.py 
      -i [prediction file]  #Training the models will create csv files containinge the predictions in the given output directory. Provide the file location of these csv files for evaluation
      -p [post-edit method] #proivde the post-edit method. Methods are described above. 
      
  ```
  
  To get the results provided in the report, try csv files from ``` line/best/results ```.
  
  ## Train neuspell model  <a name="neu"></a>
  
 ### Data Pre-Processing
 Before training, we have to pre-process the prediciton files. run ```create_dataset.py``` from the ```port/neuspell``` folder
 
  ```
    cd post/neuspell
    python3 create_dataset.py 
    -i [input directory that contains prediction files] #by default it takes best results form ```line/best/results```
    -o [desired output directory] #a directory to store formatted dataset
    -e [the line recognition model epoch] #prediction file names from the line recognition model contains epoch number that seperates different training outcomes.
    
   
    
 ```
  ### Train Model
  
To train the models, run the following code. It will save outputs in the data output directory created before

```
cd post/neuspell
python3 train -e [epochs] 
```
