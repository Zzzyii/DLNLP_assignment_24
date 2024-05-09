# %%
# Import the required libraries
import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir(os.path.split(__file__)[0])
# Setting the seaborn style
sns.set()

# Setting fonts according to the operating system to ensure the correct display of text in the charts
import platform
if platform.system() == "Windows":
    plt.rcParams['font.family'] = ['SimHei']  # For Windows systems
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # For MacOS systems
plt.rcParams['axes.unicode_minus'] = False  # Ensure that the negative sign in the chart displays correctly

data = pd.read_csv("./Dataset/suspicious tweets.csv")
data

 
# %%
# ======================================================================================================================
# Data preprocessing
# Find the minimum number of samples in the label
min_samples = data['label'].value_counts().min()

# Sample each label to ensure an equal number of samples per label
data = pd.concat([data.loc[data['label'] == label].sample(frac=1).head(min_samples) for label in data['label'].unique()]).sample(3000,random_state=42)

# Remove missing values and reset the index
data = data.dropna().reset_index(drop=True)

# Make a copy of the original data for backup
data_raw = data.copy()


# %%
# Define a function to extract English from text
def extract_english(text: str):
    # Remove non-alphabetic characters, convert to lowercase, and split
    words = text.lower().split(' ')
    words = [re.sub(r'[^a-z]', ' ', w.lower()).strip() for w in words]
    return ' '.join(words)

# %%
# data = data.sample(300)
texts = data['message'].tolist()
y = data['label'].tolist()

# %%
import tensorflow as tf
from keras.layers import TextVectorization, Embedding, MultiHeadAttention, Dropout, LayerNormalization, Dense
# Converting labels to NumPy arrays
y = np.array(y)
texts = np.array(texts)

# parameters
vocab_size = 10000  # vocabulary size
sequence_length = 100  # Maximum length of each text

# Importing tensorflow's text vectorization library
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=sequence_length)

# Using vectorizer to adapt text data
vectorizer.adapt(texts)

# Processing text data using the Vectorisation layer
text_vectorized = vectorizer(texts)


# ======================================================================================================================
## Build models
from Transformer_Model.model import build_transformer_cnn_bilstm_model
from LSTM_Model.model import build_lstm_model
from GRU_Model.model import build_gru_model
from CNN_Model.model import build_cnn_model

def get_model(model_name, vocab_size, embedding_dim,sequence_length):
    if model_name == "LSTM":
        return build_lstm_model(vocab_size, embedding_dim,sequence_length)
    elif model_name == "GRU":
        return build_gru_model(vocab_size, embedding_dim,sequence_length)
    elif model_name == "CNN":
        return build_cnn_model(vocab_size, embedding_dim,sequence_length)
    elif model_name == "Transformer":
        num_heads = 4  # Number of attention heads
        ff_dim = 32  # feedforward network dimension
        model = build_transformer_cnn_bilstm_model(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim,sequence_length=sequence_length)
        return model
    else:
        raise ValueError("Unsupported model name")
    

# %%
# ======================================================================================================================
## Print and Plot your results 

def evaluation_classification(y_test,y_pred):
    """Evaluation of classification models"""
    metrics = {}
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score  #accuracy
    from sklearn.metrics import precision_score	#precision
    from sklearn.metrics import recall_score	#recall
    from sklearn.metrics import f1_score		#F1
    # metrics['cls_report'] = classification_report(y_test, y_pred)
    print("Classified reports：")
    print(classification_report(y_test, y_pred))
    metrics['accuracy'] = accuracy_score(y_test,y_pred)
    metrics['precision'] =  precision_score(y_test,y_pred,average='macro')
    metrics['recall'] = recall_score(y_test,y_pred,average='macro')
    metrics['f1-score'] = f1_score(y_test,y_pred,average='macro')
    return metrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_roc(model,texts,y_test,name=''):
	# Calculating predictive probabilities
	y_pred_proba = model.predict(texts).ravel()
	# Calculating the ROC Curve
	fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
	roc_auc = auc(fpr, tpr)
	# Plotting the ROC Curve
	plt.figure(figsize=(8, 6))
	plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([-0.01, 1.01])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(f'ROC Curve - {name}')
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig(f"{name}_roc.jpg",dpi=300)
	plt.show()


def plot_cm(y_test,y_pred,name=''):
	# Calculate the confusion matrix
	conf_matrix = confusion_matrix(y_test, y_pred)

	# Plotting the confusion matrix
	plt.figure(figsize=(8, 6))
	sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.title(f'Confusion Matrix - {name}')
	plt.tight_layout()
	plt.savefig(f"{name}_cm.jpg",dpi=300)
	plt.show()


# %%
# ======================================================================================================================
## Training and Evaluating Models(CNN, GRU, LSTM, Transformer)

text_vectorized_np = text_vectorized.numpy()

# %%
from sklearn.model_selection import train_test_split
epochs_dic = {
    "LSTM": 2, 
    "CNN": 2,
    "GRU": 2, 
    "Transformer": 15,  
}

eval_dic = {}
eval_dic_train = {}
for model_name,epoch in epochs_dic.items():
    print(f"======{model_name}======")
    embedding_dim = 128  # Embedding layer dimensions
    # For example, get the LSTM model
    model = get_model(model_name, vocab_size=vocab_size, embedding_dim=embedding_dim,sequence_length=sequence_length)

    # Splitting the dataset
    texts_train, texts_test, y_train, y_test = train_test_split(text_vectorized_np, y, 
                                                                test_size=0.2, 
                                                                random_state=42)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Training Models
    history = model.fit(texts_train, y_train, batch_size=64, epochs=epoch, validation_data=(texts_test, y_test),verbose=0)


    # Evaluate the models on the training set
    y_pred = model.predict(texts_train)
    y_pred = (y_pred > 0.5).astype("int32")
    metric_train = evaluation_classification(y_train,y_pred)
    eval_dic[model_name] = metric_train.copy()
    print('evaluation on train：',metric_train)

    # Evaluate models on the test set
    y_pred = model.predict(texts_test)
    y_pred = (y_pred > 0.5).astype("int32")
    metric = evaluation_classification(y_test,y_pred)

    print('evaluation on test：',metric)

    plot_roc(model,texts_test,y_test,name=model_name)
    plot_cm(y_test,y_pred,name=model_name)

    eval_dic[model_name] = metric.copy()
    print("="*100)


# %%
# ======================================================================================================================
## Compare models
def plot_evaluation_comparison(eval_dic,metric_x='Metric',metric_hue='Model',dataset_name=None):
	"""
	metric_dic looks like: {'model_name':{'metric_name': value}}
	"""
	# convert to pd.dataframe
	eval_df = pd.DataFrame([[md,mt.title(),v] for md,dic in eval_dic.items() for mt,v in dic.items()],
							columns= ['Model','Metric','Value'])
	eval_df.sort_values(by=['Metric','Value'],inplace=True)
	eval_df.reset_index(drop=True,inplace=True)
	print(eval_df)
	plt.figure(figsize=(10,7))
	sns.barplot(data=eval_df,x = metric_x,y = 'Value',hue=metric_hue)
	plt.title(f"Model Comparison",fontsize=15)
	plt.xticks(rotation=0)
	plt.ylim(eval_df['Value'].min()*0.8,eval_df['Value'].max()*1.05)
	plt.legend(loc = 0, prop = {'size':8})
	plt.tight_layout()
	plt.savefig((f"{dataset_name} - " if dataset_name else '') + f"Comparison.jpg",dpi=300)
	plt.show()
	return eval_df


eval_df = plot_evaluation_comparison(eval_dic)
eval_df.to_excel("performance.xlsx",index=None)
eval_df

# %%



