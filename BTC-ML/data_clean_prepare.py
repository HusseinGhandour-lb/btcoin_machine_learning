import pandas as pd
import nltk
from nltk.corpus import stopwords

#we use nltk library and this function to common words to increase the sentence purety
nltk.download('stopworads')
stop_words = set(stopwords.words('english'))

def remove_stop_words(sentence):
    if isinstance(sentence, str):
        words = sentence.split()
        
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        return ' '.join(filtered_words)
    return sentence

#Load the data of the tweet files and the statistics file
df1 = pd.read_csv('btc_sentiment.csv')
df2 = pd.read_csv('Bitcoin-2024.csv')

#change the date column to in both dataframe to datetime inoder to set them as index and mearge the 2 dfs
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df1['Date'])

df = pd.merge(df1, df2, on='Date', how='inner')

#check for any null columns to drop them them later
na = df.isnull().sum()

#clean a set of statistics columns to change their data type(row 31 --> 42)
remove_comma_list = ['Price', 'Open', 'High', 'Low']
for column_edit in remove_comma_list:
    df[column_edit] = df[column_edit].str.replace(',','')
    df[column_edit] = pd.to_numeric(df[column_edit])
    
df['Accurate Sentiments'] = pd.to_numeric(df['Accurate Sentiments'])

df['Change %'] = df['Change %'].str.replace('%','')
df['Change %'] = pd.to_numeric(df['Change %'])

df['Vol.'] = df['Vol.'].str.replace('K','000')
df['Vol.'] = pd.to_numeric(df['Vol.'])

#create a new features to use in analysis and study the bitcoin change
df['roolin7'] = df['Price'].rolling(window=7).mean()
df['roolin30'] = df['Price'].rolling(window=30).mean()
df['shift_price'] = df['Price'].shift(1)

#edit the text columns to use in sentiment analysis ,maintaining the old column and creating a new one 
df['Short Description'] = df['Short Description'].str.lower()
symbols_remove = "1234567890@#$%^&*()_+-*\/=|.,"
tranformation = str.maketrans("", "", symbols_remove)
df['Short Description'] = df['Short Description'].str.translate(tranformation)
df['new Short Description'] = df['Short Description'].apply(remove_stop_words)

#this column is to set the score of the sentiment as negative positive or what
df['sentiment_result'] = df['Accurate Sentiments'].apply(lambda x:1 if x>0 else -1 if x<0 else 0)

#drop all the null values and create a new csv file to save the data and use it
df = df.dropna()
df.to_csv('btc_all_data.csv')
print('Done! :)')