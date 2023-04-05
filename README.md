### Table of Contents

0. [Summary](#summary)
1. [Installation](#installation)
2. [Run Predcition](#prediction)
3. [Train line recognition model](#trainOCR)
4. [Evaluate Line recognition model and post-edit methods](#eval)
5. [Train neuspell model](#neu)

## Summary  <a name="summary"></a>

With the advancement of OCR models, Text Detection from Images has made significant strides in recent years. Even while these models work exceptionally well with digital documents, there is always a potential for improvement in detecting text lines in handwritten documents. Variations in writing style, font size, and image quality have exacerbated this difficulty. There are two typical methods for detecting text lines. Segmenting lines and characters prior to employing a character recognition classifier to map characters to the final text output is one more standard approach. Using an end-to-end model to directly detect text lines from raw image inputs is another method. In this work, utilizing the popular IAM dataset, I converted handwritten line images to text output using an end-to-end approach. In addition, I investigated how the output of the end-to-end model may be further processed to improve text recognition quality. I trained an end-to-end model to recognize line images using a CNN-based model. The integration of spell correction techniques to improve the performance of this end-to-end architecture was my key focus. I have employed three unique post-processing strategies, one of which (brute force) leverages existing vocabulary to replace misspelled words, whereas the other two (edit-distance+LM and NeuSpell) use pre-trained language models to rectify spelling issues. I have concluded that brute force is the most effective tactic and have obtained a Character Error Rate (CER) of 5% using this method. 

[Full Report](https://drive.google.com/file/d/1afOm9xnK7QIiteWZch7fRdHSIabVt9te/view)

## Installation <a name="installation"></a>

Requires `python 3.6+`

Clone this repository: `git clone  https://github.com/Rowan1697/IAM.git`

To install all the libraries: `pip install -r requirements.txt`. 

To download and set up the necessary folders: ``./setup.sh``

Depending on the CUDA version, you may have to install the older versions of torch and torchvision. By default, it will install the latest version of torch and torchvision.


## Run Prediction <a name="prediction"></a>

To test the line recognition model on the IAM dataset, run

```python3 recognition.py -i [path to line image directory]```

The code will write the predicted text into a new *.txt* file with the same file name as the input line image in a new ``./results`` directory 

By default, the code generates predictions from the best OCR model along with the brute force post-edit method. This combination performs best compare to other methods. To try other post-edit methods, run:


```python3 recognition.py -i [path to line image directory] -p [provide post edit option]```

Four post edit options are available:
  - ``brute`` (replace the misspelled word with a word that needs a minimum number of edits)
  - ``candidate`` (use language model to find the best replacement of the misspelled word)
  - ``neuspell`` (a neural spell correction tool that uses a transformer-based language model to build a spell correction model)
  - ``neuspell-edit`` (only replace the misspelled word with neuspell model prediction and keep others unchanged)
  - ``no`` (no post edits to perform)
  
 Example: ```python3 recognition.py -i [path to line image directory] -p brute```
 
 To use `neuspell` post-edit methods, GPU is required. 
 
 ## Train Line recognition model <a name="trainOCR"></a>
 
 To train the Line recognition model, follow these steps:
 
 ### Data Pre-Processing
 
  Download the IAM dataset. It is expected the dataset consist of a folder with all line images and a .txt file containing respective ground truths.
  Run the dataset.py file from the main project folder. Try ``` python3 dataset.py -h ``` to know more about arguments
  ```
  python3 dataset.py 
  -i [raw input image directory] 
  -o [desired output image directory] 
  -l [file location for the label/ground truth file]
  
  ```
  This code should create three folders: train, test, and valid containing line images. A pickle will also be created to store the ground truth transcriptions. 
  
  ### Train Model
  
  To train the model run the ``` train.py ``` file from ``line`` directory. It is recommended to use gpu for training. 
 
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
      -i [prediction file]  #Training the models will create CSV files containing the predictions in the given output directory. Provide the file location of these CSV files for evaluation
      -p [post-edit method] #proivde the post-edit method. Methods are described above. 
      
  ```
  
  To get the results provided in the report, try CSV files from ``` line/best/results ```.
  
  ## Train neuspell model  <a name="neu"></a>
  
 ### Data Pre-Processing
 Before training, we have to pre-process the prediction files. run ```create_dataset.py``` from the ```port/neuspell``` folder
 
  ```
    cd post/neuspell
    python3 create_dataset.py 
    -i [input directory that contains prediction files] #by default it takes best results form ```line/best/results```
    -o [desired output directory] #a directory to store formatted dataset
    -e [the line recognition model epoch] #prediction file names from the line recognition model contains epoch number that seperates different training outcomes.
    
   
    
 ```
  ### Train Model
  
To train the models, run the following code. It will save outputs in the data output directory created before. GPU is rquired to train this model. 

```
cd post/neuspell
python3 train.py -e [epochs] 
```


## References

1. For the line recognition model, we followed the code implementation from the paper [End-to-end Handwritten Paragraph Text Recognition Using a Vertical Attention Network](https://pubmed.ncbi.nlm.nih.gov/35077353/). The GitHub repository can be found [here](https://github.com/FactoDeepLearning/VerticalAttentionOCR).
2. For fine-tuning the [NeuSpell: A Neural Spelling Correction Toolkit](https://arxiv.org/abs/2010.11085) model, we followed this GitHub [repository](https://github.com/neuspell/neuspell). 


