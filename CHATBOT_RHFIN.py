import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the text file and preprocess the data
with open("question.txt", 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)

# Preprocess each sentence in the text
def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

corpus = [preprocess(sentence) for sentence in sentences]

def get_most_relevant_sentence(question):
    question_words = preprocess(question)

    # Simple logic: Find the sentence with the most overlapping words
    max_overlap = 0
    most_relevant_sentence = ""

    for sentence in corpus:
        overlap = len(set(question_words) & set(sentence))
        if overlap > max_overlap:
            max_overlap = overlap
            most_relevant_sentence = sentence

    return most_relevant_sentence

def chatbot(question):
    most_relevant_sentence = get_most_relevant_sentence(question)
    return most_relevant_sentence

# Create a Streamlit app
def main():
    st.title("INTRANET UJAD")
    st.write("Hello! Vous êtes la bienvenue dans votre univers. Nous sommes là pour répondre à vos questions.")

    # Initialize an empty list to store questions and responses
    dialogues = st.session_state.get('dialogues', [])

    # Get the user's question
    question = st.text_input("You:")

    # Create a button to submit the question
    if st.button("Submit"):
        if question:
            response = chatbot(question)

            # Convert the list to a string before displaying
            response_str = ' '.join(response)

            # Store the question and response in the list
            dialogues.append((question, response_str))

            # Update session state
            st.session_state.dialogues = dialogues

            # Display the entire conversation without scroll bar
            conversation = "\n\n".join([f"You: {q}\nChatbot: {a}" for q, a in dialogues])
            st.markdown(conversation)

if __name__ == "__main__":
    main()
