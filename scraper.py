import requests
from bs4 import BeautifulSoup
import csv
import string

def access_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error accessing the page: {e}")
        return None

def extract_books_from_page(page_url):
    html_content = access_page(page_url)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        books = []

        book_items = soup.select('h2 a')
        for item in book_items:
            title = item.get_text(strip=True)
            book_url = f"https://www.gutenberg.org{item['href']}"
            
            books.append({
                'Title': title,
                'URL': book_url
            })
        return books
    return []

def get_book_details(book_url):
    html_content = access_page(book_url)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        details = {}
        
        table = soup.select_one('table.bibrec')
        if table:
            for row in table.find_all('tr'):
                header = row.select_one('th')
                value = row.select_one('td')
                if header and value:
                    header_text = header.get_text(strip=True)
                    value_text = value.get_text(strip=True)
                    details[header_text] = value_text
        
        return details
    return None

def save_books_to_csv(books, csv_file='gutenberg_book_deer.csv'):
    fields = ['Title', 
              'URL', 
              'Author', 
              'Illustrator', 
              'Title', 
              'Original Publication', 
              'Credits', 
              'Language', 
              'LoC Class', 
              'Subject', 
              'Category', 
              'EBook-No.', 
              'Release Date', 
              'Copyright Status', 
              'Downloads', 
              'Price']
    
    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            csv_writer = csv.DictWriter(file, fieldnames=fields)
            csv_writer.writeheader()
            
            for book in books:
                filtered_book = {field: book.get(field, '') for field in fields}
                csv_writer.writerow(filtered_book)
                
        print(f"Book information successfully saved to {csv_file}")
    except IOError as e:
        print(f"Error saving information: {e}")

def collect_books_from_gutenberg(alphabet=string.ascii_lowercase, books_per_page=None):
    letters = alphabet
    collected_books = []
    
    for letter in letters:
        page_url = f"https://www.gutenberg.org/browse/titles/{letter}?sort_order=release_date"
        print(f"Accessing: {page_url}")

        html_content = access_page(page_url)
        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            books = extract_books_from_page(page_url)
            book_ids = set()
            for book in books:
                book_id = book['URL'].split('/')[-1]
                book_ids.add(book_id)
            total_unique_books = len(book_ids)
        else:
            total_unique_books = 0

        book_ids = set()
        page_url = f"https://www.gutenberg.org/browse/titles/{letter}?sort_order=release_date"
        print(f"Accessing: {page_url}")
        books = extract_books_from_page(page_url)
        for j, book in enumerate(books):
            if books_per_page is not None and j >= books_per_page:
                break

            book_id = book['URL'].split('/')[-1]
            if book_id in book_ids:
                continue  
            
            book_ids.add(book_id) 

            book_details = get_book_details(book['URL'])
            if book_details:
                book_details['Title'] = book['Title']
                book_details['URL'] = book['URL']
                collected_books.append(book_details)
            print(f'Collection done for book {j+1} of {total_unique_books} unique books for letter {letter} ')
        
        print(f"{total_unique_books} unique books found for letter {letter}")

    save_books_to_csv(collected_books)

if __name__ == '__main__':
    collect_books_from_gutenberg() # (alphabet='abc', books_per_page=3)
