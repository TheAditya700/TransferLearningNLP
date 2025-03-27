# **Negative Transfer in Multi-Task Learning with ALBERT-Based Models**

## **Project Overview**

This project focuses on **measuring negative transfer** in **multi-task learning** using **ALBERT-based models** for **text classification** tasks. Negative transfer occurs when training on one task harms the model's performance on another, especially when tasks are dissimilar.

We analyze the effect of task similarity on model performance by evaluating different **text classification datasets** using **cosine similarity** and **KL divergence**. The primary objective is to measure how much transfer (positive or negative) occurs when fine-tuning models across tasks of varying similarity.

### **Project Goals:**
- **Investigate the impact of task similarity** on transfer learning performance.
- **Measure negative transfer** using various NLP datasets.
- **Analyze the role of task similarity** using **Cosine Similarity** and **KL Divergence**.

## **Team Members**
- **Aditya Hriday Rath** (SRN: PES1UG22AM013)
- **Aneesh Upadhya** (SRN: PES1UG22AM022)
- **Priyansh Surana** (SRN: PES1UG22AM905)

## **Datasets Used**

The following datasets were selected to test the effects of transfer learning:
- **AG News**: Multi-class topic classification.
- **SST-2**: Binary sentiment analysis.
- **Yelp Reviews**: Multi-class sentiment analysis.
- **RTE**: Natural language inference (entailment classification).
- **PAWS**: Paraphrase detection.

### **Dataset Tasks**:
- Tasks range from **sentiment classification** to **paraphrase detection**, with varying levels of similarity, providing a rich analysis of transfer effects.

## **Key Methodology**

1. **Task Similarity Measurement**:
   - **Cosine Similarity** and **Jensen-Shannon Divergence** are used to measure the similarity between tasks.
   - High similarity tasks (e.g., sentiment analysis) are expected to show **positive transfer**, while tasks with low similarity (e.g., topic classification vs. paraphrase detection) may result in **negative transfer**.

2. **Training & Evaluation**:
   - **ALBERT models** are trained independently on each task.
   - The models are then **fine-tuned** on other tasks to observe how pre-trained knowledge transfers.
   - **Evaluation metrics**: Accuracy and F1-Score are used to assess the performance of the baseline (single-task training) models versus the fine-tuned models.

3. **Fine-Tuning & Performance Analysis**:
   - Fine-tuning involves transferring knowledge from one task to another and measuring the impact on performance.
   - **Negative transfer** is observed when fine-tuning leads to **worse performance** compared to the baseline models.

## **Key Findings**
- **Task Similarity Correlates with Transfer Effects**: Tasks with higher similarity show **positive transfer**, while tasks with lower similarity show **negative transfer**.
- **RTE** experienced **strong negative transfer** when fine-tuned on dissimilar tasks, while **PAWS** and **AG News** showed minimal negative transfer.
- The relationship between task similarity and transfer effects was quantified using **Pearson and Spearman correlations**.

## **Results & Visualizations**

- **Baseline Performance**: Models trained on individual tasks were evaluated for accuracy and F1-score.
- **Fine-Tuned Performance**: The models were fine-tuned on other tasks to measure the impact of transfer.
- **Transfer Ratio**: The Transfer Ratio (TR) was calculated to quantify the effect of fine-tuning on different tasks.

### **Visualizations**:
- **Accuracy and F1-Score Comparison**: Bar charts to compare baseline and fine-tuned performance.
- **Transfer Ratio Heatmaps**: Heatmaps showing performance changes across tasks after fine-tuning.
- **Task Similarity Heatmaps**: Visualize task similarities using **Cosine Similarity** and **Jensen-Shannon Divergence**.

## **Installation & Setup**

### **Prerequisites**:
- Python 3.x
- Install necessary libraries:
  ```bash
  pip install transformers datasets evaluate scipy scikit-learn matplotlib seaborn
  ```

### **Running the Project**

To run the project locally and replicate the results, follow these steps:

1. **Clone the Repository**:
   First, clone the repository to your local machine:
   ```bash
   git clone https://github.com/YourUsername/NegativeTransferLearningNLP.git
   cd NegativeTransferLearningNLP
```

### **Install Dependencies**

To ensure that the project runs smoothly, you'll need to install all the required dependencies listed in the `requirements.txt` file. Follow these steps:

1. **Install Required Libraries**:
   Once you have cloned the repository and navigated to the project directory, install the necessary dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies using the following commands:

```bash
pip install transformers datasets evaluate scipy scikit-learn matplotlib seaborn
```

Alternatively, you can manually install the dependencies using the following commands:

```pip install transformers datasets evaluate scipy scikit-learn matplotlib seaborn```

These libraries are essential for the following purposes:
- **transformers**: Used to load and fine-tune pre-trained models like ALBERT.
- **datasets**: For loading and preprocessing NLP datasets such as AG News, SST-2, Yelp Reviews, etc.
- **evaluate**: A library to handle the evaluation of models using metrics like accuracy and F1-score.
- **scipy**: Provides functions to compute similarity metrics such as cosine similarity.
- **scikit-learn**: For machine learning utilities, including model evaluation and preprocessing.
- **matplotlib** and **seaborn**: For generating visualizations such as bar charts, heatmaps, and scatter plots.

2. **Ensure Python Compatibility**:
   This project is designed to run on **Python 3.x**. Make sure you are using a compatible version of Python. You can verify your Python version using the command:
   
   ```python --version```

   If you encounter issues or prefer to work in an isolated environment, create a virtual environment:
   
   ```python -m venv venv```
   
   ```source venv/bin/activate```  # On macOS/Linux
   
   ```venv\Scripts\activate```     # On Windows

   After activating the virtual environment, you can install the required libraries as mentioned in the previous step.

---

### **Running the Project**

After installing the dependencies and preparing your environment, you can run the project using the following steps:

1. **Clone the Repository**:
   Clone the repository to your local machine:
   
   `git clone https://github.com/TheAditya700/TransferLearningNLP.git`
   
   `cd TransferLearningNLP`

2. **Run the Jupyter Notebook**:
   The primary notebook for training, fine-tuning, and evaluating the models is `negative_transfer_analysis.ipynb`. Run the notebook using:

   - Launch Jupyter Notebook by running:
     jupyter notebook

   - In the Jupyter interface, navigate to the file `negative_transfer_analysis.ipynb` and open it.

   This notebook will:
   - **Load the datasets**: AG News, SST-2, Yelp Reviews, RTE, and PAWS.
   - **Train ALBERT models** on each dataset independently.
   - **Fine-tune models** on other tasks to observe the effects of transfer learning.
   - **Evaluate performance** (accuracy, F1-score) on both baseline and fine-tuned models.
   - **Generate visualizations** such as bar charts, heatmaps, and scatter plots to visualize task similarity and performance changes.

3. **Track Progress**:
   The training and evaluation progress will be logged in the notebook cells. You will see updates on the progress of training and evaluation, including loss metrics and accuracy/F1-score results as the cells are executed.

4. **Output Files**:
   After running the notebook, you will find the following output files saved in your directory:
   - **`baseline_performance.csv`**: Contains accuracy and F1-score for models trained on individual tasks.
   - **`fine_tune_performance.csv`**: Contains accuracy and F1-score for models after fine-tuning across tasks.
   - **Plots**: Performance comparison charts and task similarity heatmaps saved as images.

---

### **Verifying Results**

After the notebook completes execution, you should have the following files and visualizations:

- **CSV Files**:
  - **`baseline_performance.csv`**: This file contains the performance metrics for models trained on individual tasks.
  - **`fine_tune_performance.csv`**: This file contains the performance metrics for models that have been fine-tuned on different tasks.
  - 
- **Visualizations**:
  - **Accuracy and F1-Score Bar Charts**: These charts compare the performance of the baseline models and fine-tuned models.
  - **Task Similarity Heatmap**: A heatmap that visualizes task similarities based on **Cosine Similarity** and **Jensen-Shannon Divergence**.
  - **Performance Change Visualizations**: Bar charts and heatmaps showing how performance changed after fine-tuning compared to the baseline.

---

### **Conclusion**

This project investigates **negative transfer** in **multi-task learning** using **ALBERT-based models**. By measuring the similarity between tasks and fine-tuning models across them, we found that:
- **Task similarity** plays a crucial role in the success of transfer learning.
- **Negative transfer** occurs when tasks with low similarity are used for fine-tuning, leading to performance degradation.
- **Cosine similarity** and **Jensen-Shannon divergence** are useful metrics to predict transfer learning behavior.

---

### **Future Work**
- **Test other pre-trained models** like BERT, RoBERTa, and T5 to observe differences in transfer learning effects.
- Experiment with **task-specific fine-tuning strategies** to mitigate negative transfer.
- Increase the number of **training epochs** to assess the long-term impact of fine-tuning.

---

### **Contact Information**
- **Aditya Hriday Rath**: [Email](mailto:adityahr700@gmail.com)
- **Aneesh Upadhya**: [Email](mailto:aneeshupadhya234@gmail.com)
- **Priyansh Surana**: [Email](mailto:priyanshsurana1604@gmail.com)
