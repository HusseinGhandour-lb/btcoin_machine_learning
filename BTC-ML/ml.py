import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#this function loads the data and drops any function the user want to delete
def open_data(file, columns_to_drop=None):
    df = pd.read_csv(file)
    if columns_to_drop is None:
        columns_to_drop = []
    df = df.drop(columns_to_drop, axis=1)
    return df

#this function takes the x, y and trins the model then scale the data to decrease errors
def train_model_nb(df_x, df_y):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2)
    scaler = StandardScaler()
    xs_train = scaler.fit_transform(x_train)
    xs_test = scaler.transform(x_test)
    return xs_train, xs_test, y_train, y_test, scaler

#this function predicts the price of the user input if found else it predicts the x_test pricess
def model_pred_nb(model, func, input=None):
    x_train, x_test, y_train, y_test, scaler = func
    model.fit(x_train, y_train)
    if input is None:
        return model.predict(x_test)
    s_input = scaler.transform(input)
    return model.predict(s_input)

#this function is to test the model score and shows how accurate is our model
def model_score_nb(model, func):
    x_train, x_test, y_train, y_test, scaler = func
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

#this function is found to change the input text into numbers so the model can handle it
def vectorize_text(func):
    x_train, x_test, y_train, y_test = func
    vectorizer = TfidfVectorizer()
    xtf_train = vectorizer.fit_transform(x_train)
    xtf_test = vectorizer.transform(x_test)
    return xtf_train, xtf_test, y_train, y_test, vectorizer

#this function is exclusively for the text data to trian and test the data
def train_model_text(dfx, dfy):
    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2)
    return x_train, x_test, y_train, y_test

#anthour exclusively for the text user input to test the accuracy of the model
def model_score_text(model, func):
    x_train, x_test, y_train, y_test, vectorizer= func
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

#this here give the result of the input sentence if it's positive or negative for the bitcoin 
def model_pred_text(model, func, input=None):
    x_train, x_test, y_train, y_test, vectorizer = func
    model.fit(x_train, y_train)
    if input is None:
        return model.predict(x_test)
    input_vec = vectorizer.transform(input)
    return model.predict(input_vec)

#loads the clean csv file to use and the numeric features to predict the price
df_nb = open_data('btc_all_data.csv',['Unnamed: 0','Date','Short Description','Accurate Sentiments',
                                      'new Short Description','sentiment_result'])

#create a the x and y values and test the accurace of the model
x_df_nb = df_nb.drop(columns= 'Price', axis=1)
y_df_nb = df_nb['Price']
reg = linear_model.LinearRegression()

score_nb = model_score_nb(reg, train_model_nb(x_df_nb,y_df_nb))
print(score_nb)

#loads from the same cleaned data a the features needed for the sentiment analysis
df_text = open_data('btc_all_data.csv', ['Unnamed: 0','Date','Price','Open','High','Low',
                                         'Change %','Short Description','Accurate Sentiments',
                                         'Vol.','roolin7','roolin30','shift_price'])

#preapare the x and y and use them to train the model and predict an input to test it
x_df_text = df_text['new Short Description'].values.astype(str)
y_df_text = df_text['sentiment_result']
lr = linear_model.LogisticRegression()

score_text = model_score_text(lr, vectorize_text(train_model_text(x_df_text, y_df_text)))
print(score_text)

pred = model_pred_text(lr,vectorize_text(train_model_text(x_df_text,y_df_text)), ['bitcoin price may increse becuse of economic improvment'])

#decode the predicted sentiment 
if pred == 1:
    res = 'Increse in price accourding to news'
    
elif pred == -1:
    res = 'Decrease in price accourding to news'
    
else:
    res = 'Will remain the same or small change'

#print the final result
print(pred)
print(res)