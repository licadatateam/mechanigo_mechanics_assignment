from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import regex as re
import pandas as pd
from fuzzywuzzy import fuzz

class Geocoder:
    def __init__(self):
        self.driver = None
        
    def initialize_driver(self):
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--incognito")
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def query_new_address(self, address):
        """Query Google Maps with the provided address."""
        url = "https://www.google.com/maps"
        self.driver.get(url)
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.clear()
        search_box.send_keys(address)
        search_box.send_keys(Keys.RETURN)
        print(address)
        WebDriverWait(self.driver, 20).until(self.url_matches_pattern)

    def exact_match(self):
        """Check if the current URL is an exact match pattern."""
        url_pattern = r'!3d([-+]?\d+\.\d+)'
        current_url = self.driver.current_url
        return re.search(url_pattern, current_url) is not None

    def url_matches_pattern(self, driver):
        """Check if the current URL matches the Google Maps pattern."""
        url_pattern = r'@(\d+\.\d+),(\d+\.\d+),(\d+z)'
        current_url = driver.current_url
        return re.search(url_pattern, current_url) is not None

    def get_results(self, address, street_address=False):
        """Get search results from Google Maps."""
        result_i = []
        i = 0
        element = '[class^="Nv2PK"]'
        parent_elements = self.driver.find_elements(By.CSS_SELECTOR, element)

        for parent_element in parent_elements:
            a_element = parent_element.find_element(By.TAG_NAME, 'a')
            title = a_element.get_attribute('aria-label')
            link = a_element.get_attribute('href')

            child_elements = parent_element.find_elements(By.CLASS_NAME, 'W4Efsd')
            child_element_list = list()

            for child_element in child_elements:
                child_element_list.append(child_element.text)

            result = list()
            subtext = str()
            for string in child_element_list:
                words = string.split()
                for word in words:
                    if word not in result:
                        result.append(word)
            subtext = ' '.join(result)

            if street_address:
                similarity_score = fuzz.ratio(address, title)
            else:
                similarity_score = fuzz.ratio(address, subtext)

            result_i.append({'title': title, 'subtext': subtext, 'link': link, 'similarity_score': similarity_score, 'address': address, 'run': i})
            i += 1
        return result_i

    def parse_url(self, url):
        """Parse the URL to extract address, latitude, and longitude."""
        address = ''
        parts = url.split('/')
        if len(parts) > 5:
            address = parts[5].replace('+', ' ')
        lat_pattern = r'!3d([-+]?\d+\.\d+)'
        long_pattern = r'!4d([-+]?\d+\.\d+)'
        lat_matches = re.findall(lat_pattern, url)
        long_matches = re.findall(long_pattern, url)
        lat = float(lat_matches[0]) if lat_matches else 0
        long = float(long_matches[0]) if long_matches else 0
        return address, lat, long

    def geocode(self, address_query, street_address_query):
        """Geocode the provided addresses."""
        print('Working on', address_query, '/', street_address_query)
        
        url = ''
        self.query_new_address(address_query)

        try:
            results = self.get_results(address_query)
            df_review = pd.DataFrame(results)
            print(df_review)
        except:
            results = []
            df_review = pd.DataFrame()

        if self.exact_match():
            print('Exact match')
            url = self.driver.current_url
            df_review = pd.DataFrame()
        elif len(results) == 1:
            print('Partial match')
            df_review.loc[:, 'type'] = 'Partial match'
            url = results[0]['link']
        elif len(results) > 0 and df_review['similarity_score'].max() > 50:
            print('Multiple matches')
            df_review.loc[:, 'type'] = 'Multiple matches'
            prominent_results = df_review.loc[df_review['similarity_score'].max() == df_review['similarity_score']]['link']
            if len(prominent_results)>1:
                url = prominent_results[0]
            else:
                url = df_review.loc[df_review['similarity_score'].max() == df_review['similarity_score']]['link'].item()
        else:
            print('Street matches')
            if len(df_review) > 0:
                df_review.loc[:, 'type'] = 'Multiple matches'
            self.query_new_address(street_address_query)
            if self.exact_match():
                df_review_ = pd.DataFrame()
                url = self.driver.current_url
            else:
                results = self.get_results(street_address_query, street_address=True)
                print(results)
                try:
                    df_review_ = pd.DataFrame(results)
                    df_review_.loc[:, 'type'] = 'Multiple/Street matches'
                    url = df_review_.loc[df_review_['similarity_score'].max() == df_review_['similarity_score']]['link'].iloc[0]
                except:
                    df_review_ = pd.DataFrame()
            df_review = pd.concat([df_review, df_review_], axis=0)
        address, lat, long = self.parse_url(url)

        
        df_review = df_review.reset_index(drop = True)
        return df_review, address, lat, long
    
    def analyze_set(self, df, address_col = 'address_query',street_address_col='street_address_query'):
        """Geocodes a dataframe of provided addresses."""
        
        def rapide_set():
            address = 'Rapide 1 San Antonio Makati - Auto Service, Car Repair'
            lat = 14.5640785
            long = 121.0113147
            df_review = pd.DataFrame()
            return df_review, address, lat, long
        
        df_review = pd.DataFrame()
        address_list = list()
        lat_list = list()
        long_list = list()
        
        for index, row in df.iterrows():
            address_query = row[address_col]
            street_address_query = row[street_address_col]
            
            # Apply the geocode method from the Geocoder class
            if address_query == '1166 Chino Roces Avenue, Corner Estrella, Makati, 1203 in Philippines':
                df_review_, address, lat, long = rapide_set()
            else:
                df_review_, address, lat, long = self.geocode(address_query, street_address_query)
            
            # Append results to respective lists
            df_review = pd.concat([df_review,df_review_],axis=0)
            address_list.append(address)
            lat_list.append(lat)
            long_list.append(long)

            # Add the lists as new columns in the original DataFrame
        df['address'] = address_list
        df['lat'] = lat_list
        df['long'] = long_list
        
        return df,df_review
# Usage example:
# geocoder = Geocoder()
# df, address, lat, long = geocoder.geocode(address_query, street_address_query)
