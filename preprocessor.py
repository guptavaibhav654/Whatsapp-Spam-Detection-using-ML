import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df["user_message"]:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df["user"] = users
    df['message'] = messages
    df.drop(columns=["user_message"], inplace=True)

    df["only_time"] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df["day_name"] = df["date"].dt.day_name
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    return df



def clean_data(df, user_to_drop='group_notification', message_to_drop='<Media omitted>\n' ,deleted_msg="This message was deleted"):
    df = df[df['user'] != user_to_drop]
    df = df[df['message'] != message_to_drop]
    clean_data_df=df[df['message']!=deleted_msg]
    return clean_data_df



def transform_text(clean_data_df):
    ps = PorterStemmer()
    clean_data_df = clean_data_df['message'].lower()
    clean_data_df = nltk.word_tokenize(clean_data_df)
    clean_data_df = [word for word in clean_data_df if word.isalnum()]
    clean_data_df = [word for word in clean_data_df if word not in stopwords.words('english') and word not in string.punctuation]
    clean_data_df = [ps.stem(word) for word in clean_data_df]
    return " ".join(clean_data_df)


def apply_spam_classification(clean_data_df, spam_keywords_file='spam_keyword.txt'):
    # Load spam keywords
    with open(spam_keywords_file, 'r') as file:
        spam_keywords = [line.strip().lower() for line in file.readlines()]

    # Function to detect spam and identify spam keywords
    def detect_spam_keywords(message):
        message_words = set(message.lower().split())
        detected_keywords = [keyword for keyword in spam_keywords if keyword in message_words]

        # If any spam keywords are detected, return them
        if detected_keywords:
            return True, ", ".join(detected_keywords)
        else:
            return False, ""

    # Apply spam detection and keyword identification
    clean_data_df[['is_spam', 'spam_keywords']] = clean_data_df['message'].apply(lambda message: detect_spam_keywords(message)).apply(pd.Series)

    # Convert is_spam boolean to a more descriptive string if needed
    clean_data_df['is_spam'] = clean_data_df['is_spam'].map({True: 'Yes', False: 'No'})

    return clean_data_df