import re

def wordopt(text):
    # convert into lowercase
    text = text.lower()  

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+','',text)

    # Remove HTML tags
    text = re.sub(r'<.*?>','',text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d', ' ', text)

    # Remove newline characters
    text  = re.sub(r'\n', ' ', text)

    return text

