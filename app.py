import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ---- mmodel
class RNN(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out)
        return out

class ModelTrainer():
    def __init__(self, model):
        self.model = model.to(device)

    def predict(self, x_test):
        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, 1, self.model.input_size).to(device)
            output = self.model(x_test)
            _, predicted = torch.max(output[:, -1, :], 1)
            return predicted.cpu().numpy()
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))

# --- main

st.set_page_config(
    page_title="Flower Classifier App",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Quách Xuân Nam - 20020541 - IUH\n
        https://www.facebook.com/20020541.nam'''
    }
)

# model = torch.load('best_model.pth', map_location='cpu')
model = ModelTrainer(RNN(4, 128, 3))
model.load('best_model.pth')
# model.eval()

def predict(features):
    decode = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return decode[model.predict(features)[0]]
    
def input_features():
    sepal_length = st.sidebar.number_input('Sepal length', min_value=4.3, max_value=7.9, value=5.4, step=0.1)
    sepal_width = st.sidebar.number_input('Sepal width', min_value=2.0, max_value=4.4, value=3.4, step=0.1)
    petal_length = st.sidebar.number_input('Petal length', min_value=1.0, max_value=6.9, value=1.3, step=0.1)
    petal_width = st.sidebar.number_input('Petal width', min_value=0.1, max_value=2.5, value=0.2, step=0.1)
    return [sepal_length,sepal_width,petal_length,petal_width]

def main():
    st.image('images/backgound.png')
    st.markdown('<h1 style="text-align: center;">🌷 Flower Classification</h1>', unsafe_allow_html=True)
    st.markdown('---')
    data = input_features()
    pred = predict(data)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h1 style="text-align: center;">Image flower</h1>', unsafe_allow_html=True)
        st.image(f'images/{pred}.png', width=600)

    with col2:
        st.markdown('<h1 style="text-align: center;">Prediction</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="text-align: center; border: 5px solid green; ">{pred}</h1>', unsafe_allow_html=True)
    st.caption('Modify by :blue[qxnam]')
if __name__ == '__main__':
    main()