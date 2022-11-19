# Steps and Links of Table-Transformer Model.




In PubTables-1M: Towards complete table extraction from unstructured sources, authors Brandon Smock, Rohith Pesala, and Robin Abraham put forth the Table Transformer concept. To measure advancements in table extraction from unstructured documents, table structure identification, and functional analysis, the authors provide PubTables-1M, a brand-new dataset. The authors develop two DETR models known as Table Transformers, one for table identification and one for table structure recognition.

Model github repo is in : [microsoft/table-transformer: Model training and evaluation code for our dataset PubTables-1M, developed to support the task of table extraction from unstructured documents.](https://github.com/microsoft/table-transformer)

- It contains two folder detr and src where we have to use src folder to train for both Table Detection and table Structure Recognition. Where the table detection code will detect the whole table with confidence label and with the coordinates of the bounding box around the table. And each rows and columns for the Table structure with the confidence label and bounding box for each row and column, altogether it will show a comprehensive structure.

- And the detr folder contains the detectron model which will help us to detect the tables, which acts as a wrapper for this object detection model.

It has 575,305 pages of annotated documents that have tables on them for table detection. 947,642 completely annotated tables with comprehensive location (bounding box) data and text content for identifying table structure and performing functional analysis. All table rows, columns, and cells (including empty cells), as well as other marked structures like column headers and projected row headers, have complete bounding boxes in image and PDF coordinates.

All tables and pages were rendered as pictures. For every word that appears in each table and page graphic, there are bounding boxes and text. Additional cell attributes weren't employed in the model's training at this time. To ensure the annotations are as noise-free as feasible, they had also canonicalized cells in the headers and used a number of quality check procedures. For further details you can check in there [given paper](https://arxiv.org/pdf/2110.00061.pdf)

- Also they had provided the weights of the pretrained models for both Table detection and Table structure detection which can be used to train on our new data.

- They trained the models for 20 epochs only.


You can change the number of epochs in the structure.config and dectron.config file at the src folder. And also can reduce the size of the data at the text files of the table structure and pdf annotation dataset folder for training purpose.

For the evaluation metrices used in the table transformer researcher had focused on the [GriTs method](https://arxiv.org/abs/2203.12555)

To download the dataset please follow the following link : [Microsoft Research Open Data](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3)

### Steps to Train and Test the Model.

#### First we need to create the environment if it’s not there in the conda environment list. The environment name will be tables-detr. 

> Step 1 : Check whether the env is there or not by using cmd in linux terminal :
‘conda env list ‘ where env name will be : tables-detr
If the environment is not installed then we have to create the environment by going to this particular path which is : cd SageMaker/ table-transformer/
Then run this command to create the environment : conda env create -f environment.yml

> Step 2 : Then activate the env using : source activate tables-detr

> Step 3 : After activation go to the folder where main.py is present :
cd SageMaker/table-transformer/src/

> Step 4 : Then run this command to train the table structure model : 
(tables-detr2) C:\Users\malvi\ML_project\table-transformer-main\src>python main.py --data_type structure --config_file structure_config.json --data_root_dir C://Users//malvi//Desktop//Main//PhD//course_work//ML//Project//data//pubTable//PubTables-1M-Image_Table_Structure_PASCAL_VOC//PubTables1M-Structure-PASCAL-VOC

And Then run this command to train the table detection model : 

> (tables-detr2) C:\Users\malvi\ML_project\table-transformer-main\src>python main.py --data_type detection --config_file detection_config.json --data_root_dir C://Users//malvi//Desktop//Main//PhD//course_work//ML//Project//data//pubTable//PubTables-1M-Image_Page_Detection_PASCAL_VOC//PubTables1M-Detection-PASCAL-VOC 

> Then the training will start and the model weights will get store in the output folder inside the PubTables1M-Structure-PASCAL-VOC folder. We can use this model in the inference stage to test on top of our new data.

### Huggingface Table-Transformer.

Hugging face has inferenced the Table Transformer model by using the pre-trained models for both Table Detection and Table Structure recognition. 

>	Two models were published by the authors: one for document table detection and the other for table structure recognition (the task of recognising the individual rows, columns etc. in a table).
>	In order to prepare photos and optional targets for the model, one can utilize the AutoFeatureExtractor API (Auto Classes (huggingface.co)). A DetrFeatureExtractor (DETR (huggingface.co)) will be loaded in the background as a result.


#### Table Transformer Model


![TableTransformerModel](https://github.com/AnkitMalviya/ML_Proj_2022_Phd/tree/main/assets/Picture1.jpg)


Table Transformer For Object Detection
![TableTransformerForObjectDetection](https://github.com/AnkitMalviya/ML_Proj_2022_Phd/tree/main/assets/Picture2.jpg)
 

#### Table Detection Output

![TableDetectionOutput](https://github.com/AnkitMalviya/ML_Proj_2022_Phd/tree/main/assets/Picture3.jpg)
 

![Table Detection Result](https://github.com/AnkitMalviya/ML_Proj_2022_Phd/tree/main/assets/Picture3.jpg)
 



Table Structure Recognition Output

 


 
Number of Tables and Columns
 

Confidence score and Bounding Boxes Coordinates
 




# Steps and Links of TableNet Model

With the increase use of mobile devices, customers tend to share documents as images rather than scanning them. These images are later processed manually to get important information stored in Tables. These tables can be of different sizes and structures. It is therefore expensive to get information in Tables from images.
With TableNet we will employ and end-to-end Deep learning architecture which will not only localize the Table in an image, but will also generate structure of Table by segmenting columns in that Table.
We will use both Marmot and Marmot Extended dataset for Table Recognition. Marmot dataset contains Table bounding box coordinates and extended version of this dataset contains Column bounding box coordinates.
Marmot Dataset : https://www.icst.pku.edu.cn/cpdp/docs/20190424190300041510.zip 
Marmot Extended dataset : https://drive.google.com/drive/folders/1QZiv5RKe3xlOBdTzuTVuYRxixemVIODp
Download processed Marmot dataset: https://drive.google.com/file/d/1irIm19B58-o92IbD9b5qd6k3F31pqp1o/view?usp=sharing
Model GitHub repo is in:
GitHub - asagar60/TableNet-pytorch: Pytorch Implementation of TableNet
This github link contains all the links for Downloading the Marmot Datasets which we are using to run our code. This link also contain links for medium vlogs for this research article. You will also get link for saved pretrained model (DenseNet121) which we have use to train our model.
•	Training folder you can find all the scripts which will run on background.
•	You have to  run EDA-v1 python file in the link which you have to run for EDA analysis and for generating processed_data.csv which contain Image path, table mask path, column mask path , table and column boundary boxes.
•	After that you have to run Output python file for training and testing your model.
We have around 994 images documents from which we will be dealt using semantic segmentation by predicting pixel-wise regions of Table and columns in them.
Image data is in .bmp (bitmap image file) format and bounding box coordinates are in XML files following Pascal VOC format.
 
                                  Steps for Data Pre-Processing
Image data is in .bmp (bitmap image file) format and bounding box coordinates are in XML files following Pascal VOC format.
First we define 3 utility functions
•	get_table_bbox() : This function will extract Table Coordinates using xml file from original marmot dataset and scale them w.r.t to new image shape
•	get_col_bbox() : This function will extract Column Coordinates using xml file from extended marmot dataset and scale them w.r.t to new image shape, and if no table coordinates are returned from get_table_bbox() function, we will approximate them using column bounding boxes.
•	create_mask() : This function takes in bounding boxes ( table / column) and creates mask with 1 channel. If no bounding boxes are given, it creates an empty mask.
Basic idea of preprocessing:
•	Read image file, table_xml and column_xml.
•	Resize image to (1024, 1024) and convert them to RGB ( if not already)
•	Get both table and column bounding box
•	Create mask for both
•	Save Image and mask to disk
•	converting processed_data to csv file.
	Let’s check the masks that were created based on table and column coordinates
 

                               Steps to Train and Test the Model
We will use DenseNet121 as encoder and build model upon it.
 
Path-  "/home/ec2-user/SageMaker/Ayush/TableNet-pytorch-main “
Trainable Params
 
We constrained the training epochs to 50–100, and tried different models for encoders.
 Densenet121 worked best as encoder compared to VGG19, ResNet-18 and EfficientNet. It is worth mentioning that performance of ResNet-18 and EfficientNet was almost close to DenseNet, but I chose the model based on Best F1 Score on Test data.
Test function takes data loader, model and loss as input and returns F1 Score, Accuracy, Precision, Recall and Loss for that epoch.
 
 
Model testing result: - Model testing will generate the table and column mask, using this masks we can crop the table image and then we can extract the information using Tesseract OCR, which provided by AWS.

 
After this we will crop the mask image using the boundary boxes.
 
Once we have Crop image, we can extract the information using Tesseract OCR, which provided by AWS.
 
Observations
•	we have observed   that Bad / worst predictions are given by images with colored tables. Model didn't predict anything and F1 score is close to 0.0. There are very few images in the dataset which have colored tables.
•	Good predictions come from those images which predicted good Table mask, but it also predicted columns in the table where in actual there were no columns.
•	Best Predictions are images which helped model learn table and column boundaries even without line demarcations
Fixing Image Problems
We have 2 options, which might improve model performance,
•	Remove colored images, or [ Problem: Data reduction is an issue here as we already have less data]
•	We can have uniform data by converting all images to grayscale first and then increase the number of channels in preprocessing, and Train model again.
Improving model predictions using OpenCV2
We can still see uneven boundaries of predicted table and column masks. In some cases, Table mask predictions are not even filled inside. If we directly crop the mask portions of the image to get Table, we might lose some information. Not to mention, there are other areas with activations in the predicted table mask (which are not tables).
To solve these issues, we will use contours from classical image processing techniques.
Basic Idea :
•	Get contours around the activation from the predicted table mask.
•	Remove contours which cant be rectangle / small patch of activations.
•	Get bounding coordinates of the remaining contour.
•	Repeat the same process with Column Masks
Reference:- [2001.01469] TableNet: Deep Learning model for end-to-end Table detection and Tabular data extraction from Scanned Document Images (arxiv.org)


















