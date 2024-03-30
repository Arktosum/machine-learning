import os

def concatenate_text_files(folder_path):
    giant_string = ""
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Check if the file is a text file
            file_path = os.path.join(folder_path, filename)
            # Read the content of the text file and concatenate to the giant string
            with open(file_path, 'r', encoding='utf-8') as file:
                giant_string += file.read()
    return giant_string
import re

def remove_special_characters(word):
    # Define a regular expression pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # Matches any character that is not alphanumeric or whitespace
    # Replace special characters with an empty string
    clean_word = re.sub(pattern, '', word)
    return clean_word



def processCORPUS(corpus):
    freq = {}
    for word in corpus:
        for letter in list(word):
            if letter not in freq:
                freq[lett  er]=0
            freq[letter]+=1
    print(list(set(corpus)))
        

corpus = []
def main():
    folder_path = './wiki-pages'  
    giant_string = concatenate_text_files(folder_path)
    
    
    for line in giant_string.split("\n"):
        if line and not line.startswith('=='): 
            for sentence in line.split("."):
                for word in sentence.split(" "):
                    WORD = remove_special_characters(word.lower())
                    if WORD:
                        corpus.append(WORD)

    
    processCORPUS(corpus)
if __name__ == "__main__":
    main()