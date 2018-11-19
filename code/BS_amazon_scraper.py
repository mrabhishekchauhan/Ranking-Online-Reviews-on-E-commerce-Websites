from bs4 import BeautifulSoup
import requests
import csv
pdt_code = "B00BSE5WQ4"
csv_file = open('Fastrack_Black_Magic_Analog_Black_Dial_Mens_Watch.csv', 'w', encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Product_Id','Rating','Review_Text','Helpfulness','B_Helpfulness'])
for z in range(1,25):
    url = "https://www.amazon.in/Fastrack-Black-Magic-Analog-Watch/product-reviews/B00BSE5WQ4/ref=cm_cr_getr_d_paging_btm_"+(str)(z)+"?ie=UTF8&reviewerType=all_reviews&pageNumber="+(str)(z)+""
    source = requests.get(url).text
    soup = BeautifulSoup(source, 'lxml')
    for r in soup.findAll(class_='a-section celwidget'):
        review_rating = r.find('div', class_='a-row')
        numeric_review_rating = review_rating.find(class_='a-link-normal')
        numeric_review_rating = review_rating.find(class_='a-link-normal')['title']
        numeric_review_rating = review_rating.find(class_='a-link-normal')['title'].split(' ')[0]
        final_numeric_review_rating = numeric_review_rating.split('.')[0]
        print(pdt_code)
        print(final_numeric_review_rating)
        review_text = r.find('span', class_='a-size-base review-text').text
        print(review_text)
        helpfulness = r.find('span', class_='cr-vote-buttons cr-vote-component')
        helpfulness = helpfulness.find('span', class_='a-color-secondary').text
        review_helpfulness = helpfulness.split('\n')
        len_review = len(review_helpfulness)
        if (len_review == 1):
            hp = 0
            bhp = 0
            print(hp,bhp)
        else:
            bhp = 1
            semifinal_helpfulness = review_helpfulness[1]
            final_helpfulness = semifinal_helpfulness.split(' ')[6]
            if (final_helpfulness == 'One'):
                hp = 1
            else:
                hp = (final_helpfulness)
            print(hp,bhp)
        csv_writer.writerow([pdt_code, final_numeric_review_rating, review_text, hp, bhp])
        print()