from bs4 import BeautifulSoup as BS
from selenium import webdriver
from functools import reduce
import pandas as pd
import time
import matplotlib.pyplot as plt
from selenium.webdriver.firefox.options import Options
import numpy as np

options = Options()
options.add_argument('--headless')

def render_page(url):
        driver = webdriver.Firefox(options=options)
        driver.get(url)
        time.sleep(1)
        r = driver.page_source
        driver.quit()
        return r


class WUscrape:
    def __init__(self, city, year):
        self.city = city
        self.year = year

    def city_find(self):

        if self.city == 'Zagreb':
            string = 'LDZA'
        elif self.city == 'Osijek':
            string = 'LDOS'
        elif self.city == 'Split':
            string = 'LDSP'
        elif self.city == 'Rijeka':
            string = 'LDRI'
        elif self.city == 'Dubrovnik':
            string = 'LDDU'
        else:
            print('City is not in the database!')
            return

        return string 

    def scrape(self, city_string):
        page = 'https://www.wunderground.com/history/daily/' + city_string + '/date/'
        temp = np.array([])

        year_str = str(self.year)

        for m in range(12):
            m += 1

            m_str = f'-{m}-'
            
            d31 = [1, 3, 5, 7, 8, 10, 12]
            d30 = [4, 6, 9, 11]

            t_month = []
            d = 0

            if m in d31:
                while d < 31:
                    d += 1
                    d_str = f'{d}'

                    url = page + year_str + m_str + d_str
                    r = render_page(url)

                    try:
                        soup = BS(r, "html.parser")
                        container = soup.find('table', class_='mat-table cdk-table mat-sort ng-star-inserted')
                        container = container.find_all('span', class_='test-true wu-unit wu-unit-temperature is-degree-visible ng-star-inserted')
                        for i in range(len(container)):
                            check = container[i].find('span', class_='wu-value wu-value-to')
                            t_month.append((float(check.text) - 32) * 5 / 9)
                        print(f'Done for: month {m} - day {d}')
                    except:
                        d = d-1
                        print(f'Reruning for: month {m} - day {d+1}')
                 
                temp = np.append(temp, t_month)
                #plt.plot(temp)
                #plt.show()

            elif m in d30:
                while d < 30:
                    d += 1
                    d_str = f'{d}'

                    url = page + year_str + m_str + d_str
                    r = render_page(url)

                    try:
                        soup = BS(r, "html.parser")
                        container = soup.find('table', class_='mat-table cdk-table mat-sort ng-star-inserted')
                        container = container.find_all('span', class_='test-true wu-unit wu-unit-temperature is-degree-visible ng-star-inserted')
                        for i in range(len(container)):
                            check = container[i].find('span', class_='wu-value wu-value-to')
                            t_month.append((float(check.text) - 32) * 5 / 9)
                        print(f'Done for: month {m} - day {d}')
                    except:
                        d = d-1
                        print(f'Reruning for: month {m} - day {d}')
                 
                temp = np.append(temp, t_month)
                #plt.plot(temp)
                #plt.show()

            elif m == 2:
                if self.year % 4 == 0:
                    while d < 29:
                        d += 1
                        d_str = f'{d}'

                        url = page + year_str + m_str + d_str
                        r = render_page(url)

                        try:
                            soup = BS(r, "html.parser")
                            container = soup.find('table', class_='mat-table cdk-table mat-sort ng-star-inserted')
                            container = container.find_all('span', class_='test-true wu-unit wu-unit-temperature is-degree-visible ng-star-inserted')
                            for i in range(len(container)):
                                check = container[i].find('span', class_='wu-value wu-value-to')
                                t_month.append((float(check.text) - 32) * 5 / 9)
                            print(f'Done for: month {m} - day {d}')
                        except:
                            d = d-1
                            print(f'Reruning for: month {m} - day {d}')
                    
                    temp = np.append(temp, t_month)
                    #plt.plot(temp)
                    #plt.show()
                    
                else:
                    while d < 28:
                        d += 1
                        d_str = f'{d}'

                        url = page + year_str + m_str + d_str
                        r = render_page(url)

                        try:
                            soup = BS(r, "html.parser")
                            container = soup.find('table', class_='mat-table cdk-table mat-sort ng-star-inserted')
                            container = container.find_all('span', class_='test-true wu-unit wu-unit-temperature is-degree-visible ng-star-inserted')
                            for i in range(len(container)):
                                check = container[i].find('span', class_='wu-value wu-value-to')
                                t_month.append((float(check.text) - 32) * 5 / 9)
                            print(f'Done for: month {m} - day {d}')
                        except:
                            d = d-1
                            print(f'Reruning for: month {m} - day {d}')
                    
                    temp = np.append(temp, t_month)
                    #plt.plot(temp)
                    #plt.show()
            
            np.savetxt('data/raw/' + self.city + year_str + '.csv', temp, delimiter=",")

        return temp
            