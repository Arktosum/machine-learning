import wikipedia

def fetch_wikipedia_content(query):
    try:
        search_results = wikipedia.search(query)
        for result in search_results:
            try:
                page = wikipedia.page(result)
                save_to_txt(result,page.content)
            except wikipedia.exceptions.PageError:
                print(f"Page related to '{result}' not found.")
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation: {e.options}")
        print("No valid search results found.")
        return None
    except wikipedia.exceptions.WikipediaException as e:
        print(f"Wikipedia Exception: {e}")
        return None
    
def save_to_txt(filename,content):
    with open(filename+'.txt', 'w', encoding='utf-8') as file:
        file.write(content)

pages = ['science','technology','space','nietczhe']
def main():
    for page in pages:
        fetch_wikipedia_content(page)

if __name__ == "__main__":
    main()
