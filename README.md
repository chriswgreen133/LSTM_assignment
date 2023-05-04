# LSTM - Assignment
<hr />
Follow the below steps to run to scripts.

## Step 1:
Download and Install Miniconda, follow installation instructions from below website

``` https://docs.conda.io/en/latest/miniconda.html ```

## Step 2:
Create new Conda environment by running below commands in your terminal

``` 
    conda create --name lstm python=3.8
    conda activate lstm
```

## Step 3:
Install required packages by navigating to the project directory and running below line in your terminal

``` pip install -r requirements.txt ```

## Step 3:
Download dataset by running below command in terminal:

#### Windows
``` 
    curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip 
    tar -xf household_power_consumption.zip
```

#### Linux/Mac
``` 
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip 
    unzip household_power_consumption.zip
```


## Step 5:
Run the LSTM_train.py Script by running below command in terminal

``` python LSTM.py ```
