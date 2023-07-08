import secrets
import hashlib
from flask_caching import Cache
import re
import time
from flask import Flask, flash, render_template, request
import requests
import pandas as pd
from urllib.parse import quote
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from selenium.common.exceptions import ElementNotInteractableException
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver import ChromeOptions, Chrome
from selenium.webdriver.common.keys import Keys
import math

from itertools import chain
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import joblib
import spacy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

opts = ChromeOptions()
opts.add_experimental_option("detach", True)
opts.add_experimental_option('excludeSwitches', ['enable-logging'])
# opts.add_argument("--headless")
driver = webdriver.Chrome(executable_path='E:/chromedriver.exe')



app = Flask(__name__)
app = Flask(__name__, static_folder='templates\static')
cache = Cache(app)

secret_key = secrets.token_hex(16)

app.config['SECRET_KEY'] = secret_key


@app.route('/')
def logins():
    return render_template('login.html')


@app.route('/laptop_category')
def laptop_category():
    return render_template("laptop_categories.html")


@app.route('/tv_category')
def tv_category():
    return render_template("tv_categories.html")


@app.route("/mobile_category")
def mobile_category():
    return render_template("mobile_categories.html")


@app.route('/smartwatch_category')
def smartwatch_category():
    return render_template("smartwatch_categories.html")


@app.route('/firebolt_watch', methods=['GET'])
def firebolt_watch():
    print("hello")
    flash('Please Wait till products are fetched!')

    fir = amazon_firebolt_watch_prod()
    return render_template("firebolt_watches.html", fir=fir)


@app.route('/boat_watch', methods=['GET'])
def boat_watch():
    print("hello")
    flash('Please Wait till products are fetched!')

    bor = amazon_boat_watch_prod()
    return render_template("boat_watches.html", bor=bor)


@app.route('/apple_watch', methods=['GET'])
def apple_watch():
    print("hello")
    flash('Please Wait till products are fetched!')

    appler2 = amazon_apple_watch_prod()
    return render_template("apple_watches.html", appler2=appler2)


@app.route('/hp_laptop', methods=['GET'])
def hp_laptop():
    print("hello")
    flash('Please Wait till products are fetched!')

    hpr = amazon_hp_laptop_prod()
    return render_template("hp_laptop.html", hpr=hpr)


@app.route('/asus_laptop', methods=['GET'])
def asus_laptop():
    print("hello")
    flash('Please Wait till products are fetched!')

    asusr = amazon_asus_laptop_prod()
    return render_template("asus_laptop.html", asusr=asusr)


@app.route('/dell_laptop', methods=['GET'])
def dell_laptop():
    print("hello")
    flash('Please Wait till products are fetched!')

    dellr = amazon_dell_laptop_prod()
    return render_template("dell_laptop.html", dellr=dellr)


@app.route('/lg_tv', methods=['GET'])
def lg_tv():
    print("hello")
    flash('Please Wait till products are fetched!')

    lgr = amazon_lg_tv()
    return render_template("lg_tv.html", lgr=lgr)


def amazon_lg_tv():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "lg tv", Keys.RETURN)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_lg_tv_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_lg_tv_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)

            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


@app.route('/sony_tv', methods=['GET'])
def sony_tv():
    print("hello")
    flash('Please Wait till products are fetched!')

    sonyr = amazon_sony_tv()
    return render_template("sony_tv.html", sonyr=sonyr)


@app.route('/samsung_tv', methods=['GET'])
def samsung_tv():
    print("hello")
    flash('Please Wait till products are fetched!')

    samsungr = amazon_samsung_tv_prod()
    return render_template("samsung_tv.html", samsungr=samsungr)


@app.route('/apple_mobile', methods=['GET'])
def apple_mobile():
    print("hello")
    flash('Please Wait till products are fetched!')

    appler = amazon_apple_mobile_prod()
    return render_template("apple_mobiles.html", appler=appler)


@app.route('/mi_mobile', methods=['GET'])
def mi_mobile():
    print("hello")
    flash('Please Wait till products are fetched!')

    mir = amazon_mi_mobile_prod()
    return render_template("mi_mobiles.html", mir=mir)


@app.route('/samsung_mobile', methods=['GET'])
def samsung_mobile():
    print("hello")
    flash('Please Wait till products are fetched!')

    samsungr = amazon_samsung_mobile_prod()
    return render_template("samsung_mobiles.html", samsungr=samsungr)


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    return render_template('search.html')


timestamps = []
execution_times = []


def time_graph():
    start_time = time.time()

    # Code execution points
    # Measure execution time at different points in your code
    # Example:
    for i in range(10):
        time.sleep(1)  # Simulate some processing time
        timestamp = time.time() - start_time
        timestamps.append(timestamp)

    # Calculate execution time
    execution_time = time.time() - start_time
    execution_times.append(execution_time)


@app.route('/searchs')
def search_results():
    time_graph()
    query = request.args.get('query')
    print(query)

    lresults = amazon_products(query)
    time.sleep(10)
    lresults2 = flipkart_products(query)

    return render_template('search_results.html',  lresults=lresults, lresults2=lresults2)


def amazon_sony_tv():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "sony tv", Keys.RETURN)
    WebDriverWait(driver, 20)
    WebDriverWait(driver, 20)

    WebDriverWait(driver, 20)
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_sony_tv_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_sony_tv_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 10:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1

            except:
                print("no ratings found cannot predict")
        print(lresults)

        return lresults


def amazon_boat_watch_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "boat smartwatch", Keys.RETURN)
    WebDriverWait(driver, 20)
    WebDriverWait(driver, 20)
    driver.find_element(
        By.XPATH, "//ul//li//span//a//span[contains(text(),'boAt')]").click()
    WebDriverWait(driver, 20)
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_boat_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_boat_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_apple_watch_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "apple smartwatch", Keys.RETURN)
    WebDriverWait(driver, 20)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_apple_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_apple_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_firebolt_watch_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "firebolt smart watch", Keys.RETURN)
    WebDriverWait(driver, 20)

    WebDriverWait(driver, 20)
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_firebolt_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_firebolt_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_hp_laptop_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "Hp laptops", Keys.RETURN)
    WebDriverWait(driver, 20)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_hp_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_hp_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                time.sleep(15)
                driver.get(item)

                time.sleep(10)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_dell_laptop_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "Dell laptops", Keys.RETURN)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_dell_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_dell_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 8:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)

                WebDriverWait(driver, 10)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
        print(lresults)

        return lresults


def amazon_asus_laptop_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "asus laptops", Keys.RETURN)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_asus_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_asus_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_samsung_tv_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "Samsung tv", Keys.RETURN)
    WebDriverWait(driver, 20)

    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:

        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_samsung_tv_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_samsung_tv_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
        print(lresults)

        return lresults


def amazon_samsung_mobile_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "Samsung Smartphone", Keys.RETURN)
    WebDriverWait(driver, 20)

    WebDriverWait(driver, 20)
    links = WebDriverWait(driver, 20).until(EC.presence_of_element_located(
        (By.XPATH, "//div//div//span//h2//a[contains(@class, 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")))
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")

    productLinks = []

    for link in links:

        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_samsung_mobile_product_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_samsung_mobile_product_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)

            try:
                WebDriverWait(driver, 10)
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_apple_mobile_prod():
    print("hai")
    driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "Apple iphone", Keys.RETURN)
    WebDriverWait(driver, 20)

    WebDriverWait(driver, 20)
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_apple_mobile_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_apple_mobile_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


def amazon_mi_mobile_prod():
    print("hai")
    try:
        driver.get("https://www.amazon.in")
    finally:
        driver.get("https://www.amazon.in")

    WebDriverWait(driver, 10)

    driver.find_element(By.ID, 'twotabsearchtextbox').send_keys(
        "mi smartphone", Keys.RETURN)
    WebDriverWait(driver, 20)

    WebDriverWait(driver, 20)
    links = driver.find_elements(
        By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
    productLinks = []

    for link in links:
        href = link.get_attribute("href")
        productLinks.append(href)

    csv_header = 'product_link\n'

    csv_rows = [f"{link}\n" for link in productLinks]
    csv_data = csv_header + ''.join(csv_rows)
    with open('amazon_mi_links.csv', 'w') as file:
        file.write(csv_data)
    results_data = []
    res = []
    lresults = []
    with open('amazon_mi_links.csv', 'r') as csvfile:

        reader = csv.reader(csvfile)

        next(reader)
        counter = 0

        for row in reader:
            for item in row:
                if counter >= 12:
                    break

                WebDriverWait(driver, 10)
                driver.get(item)
                WebDriverWait(driver, 20)
            try:
                pname = driver.find_element(
                    By.ID, "productTitle").text
                pprice = driver.find_element(
                    By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                productImage = driver.find_element(
                    By.CSS_SELECTOR, "#imgTagWrapperId img")
                imageSource = productImage.get_attribute("src")
                ratings = driver.find_element(
                    By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                reviews = driver.find_elements(
                    By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
            except:
                pname = "product name not found"
                pprice = "price not found"
                productImage = "product image not found"
                imageSource = "product image not found"
                ratings = "ratings not found"
                reviews = "no reviews"
            try:
                product_rev = ""
                pr = []
                for revi in reviews:
                    revi_rev = revi.text
                    pr.append(revi_rev)

                for i in pr:

                    loaded_rf = joblib.load("model.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer = joblib.load(
                        'vectorizer.joblib')
                    test_text_vectorized = loaded_vectorizer.transform(
                        test_text_preprocessed)

                    predicted_ratings = loaded_rf.predict(
                        test_text_vectorized)
                    pr_list = []
                    pr_list.append(predicted_ratings)

                '''linear regression'''
                for i in pr:
                    loaded_rf2 = joblib.load("model2.joblib")

                    test_text_preprocessed = [
                        preprocess_text(text) for text in [i]]

                    loaded_vectorizer2 = joblib.load(
                        'vectorizer2.joblib')
                    test_text_vectorized2 = loaded_vectorizer2.transform(
                        test_text_preprocessed)

                    predicted_ratings2 = loaded_rf2.predict(
                        test_text_vectorized2)
                    pr_list2 = []
                    pr_list2.append(predicted_ratings2)

                average2 = sum(pr_list2) / len(pr_list2)
                av = abs(average2-1)*1

                average_linear = np.array2string(
                    av, separator=', ')

                average = sum(pr_list) / len(pr_list)
                average_random = np.array2string(
                    average, separator=', ')

                results = {}
                results["product_name"] = pname
                results['product_price'] = pprice
                results["image_source"] = imageSource
                results["amazon_ratings"] = ratings
                results["random_forest"] = average_random
                results["linearRegression"] = average_linear
                results["product_link"] = item

                lresults.append(results)

                counter += 1
            except:
                print("no ratings found cannot predict")
    print(lresults)

    return lresults


@app.route('/signup_lo', methods=['POST'])
def signup_lo():

    salt = 'randomsalt'
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    hash_object = hashlib.sha256((password + salt).encode('utf-8'))
    hashed_password = hash_object.hexdigest()

    with open('users.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, email, hashed_password])

        return render_template('login.html')


@app.route('/login_si', methods=['POST'])
def login_si():
    email = request.form['email']
    password = request.form['password']
    salt = 'randomsalt'

    hash_object = hashlib.sha256((password + salt).encode('utf-8'))
    entered_hash = hash_object.hexdigest()
    with open('users.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if email == row[1] and entered_hash == row[2]:
                return render_template("index.html")

    return render_template("index.html")


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(str(text).lower())

    # Remove stop words and punctuation
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token not in stop_words]

    # Stem the tokens
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in stemmed_tokens]

    # Return the preprocessed text as a string
    return ' '.join(lemmatized_tokens)


def pos_tag_text(text):
    # Tokenize the text
    tokens = (str(text).lower())
    # Perform part-of-speech tagging on the tokens
    pos_tags = pos_tag(tokens)

    # Return the part-of-speech tags as a list of tuples
    return pos_tags


def ner_text(text):
    # Tokenize the text
    tokens = word_tokenize(str(text).lower())

    # Perform named entity recognition on the tokens
    ne_tags = ne_chunk(pos_tag(tokens))

    # Return the named entities as a list of tuples
    return ne_tags


vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

dat_re = []
pattern_tv = re.compile(r'tv|television|Tele?visions?|TV', re.IGNORECASE)

pattern_mobile = re.compile(
    r'mobile(s)?|smartphone|Smart?phones?|Mobiles|iphone(s)?', re.IGNORECASE)
pattern_smartwatches = re.compile(r'smart\s?watch(es)?', re.IGNORECASE)
pattern_laptops = re.compile(r'laptops?', re.IGNORECASE)

pattern_hp = re.compile(r'hp|hewlett[-\s]?packard', re.IGNORECASE)
pattern_dell = re.compile(r'dell', re.IGNORECASE)
pattern_asus = re.compile(r'asus', re.IGNORECASE)
pattern_apple = re.compile(r'apple|iphone(s)?', re.IGNORECASE)
pattern_boat = re.compile(r'boat', re.IGNORECASE)
pattern_firebolt = re.compile(r'fireboltt', re.IGNORECASE)
pattern_lg = re.compile(r'lucky goldstar|lg', re.IGNORECASE)
pattern_samsung = re.compile(r'Samsung', re.IGNORECASE)
pattern_sony = re.compile(r'Sony', re.IGNORECASE)
pattern_mi = re.compile(r'Mi|Redmi|Xiaomi', re.IGNORECASE)
pattern_noise = re.compile(r'noise', re.IGNORECASE)

query = ''
input = ''


def flipkart_products(input):

    if(re.search(pattern_tv, input)):
        driver.get("https://www.flipkart.com")
        button_close = driver.find_element(By.CLASS_NAME, '_2doB4z')
        WebDriverWait(driver, 20)
        button_close.click()
        WebDriverWait(driver, 20)

        driver.find_element(
            By.CLASS_NAME, '_3704LK').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_samsung, input4)):

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            product_links = [link.get_attribute("href") for link in links]
            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_lg, input4)):
            print(input4)

            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'LG')]")))

            driver.find_element(
                By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'LG')]").click()

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

    if(re.search(pattern_mobile, input)):
        driver.get("https://www.flipkart.com")

        button_close = driver.find_element(By.CLASS_NAME, '_2doB4z')
        WebDriverWait(driver, 20)
        button_close.click()
        WebDriverWait(driver, 20)

        driver.find_element(
            By.CLASS_NAME, '_3704LK').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_samsung, input4)):

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            product_links = [link.get_attribute("href") for link in links]
            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')
                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_apple, input4)):
            print(input4)

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_mi, input4)):
            print(input4)

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

    if(re.search(pattern_laptops, input)):
        driver.get("https://www.flipkart.com")
        button_close = driver.find_element(By.CLASS_NAME, '_2doB4z')
        WebDriverWait(driver, 20)
        button_close.click()
        WebDriverWait(driver, 20)

        driver.find_element(
            By.CLASS_NAME, '_3704LK').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_hp, input4)):

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            product_links = [link.get_attribute("href") for link in links]
            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_dell, input4)):
            print(input4)
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div._203eRC._2ssEMF")))

            driver.find_element(By.CSS_SELECTOR, "div._203eRC._2ssEMF").click()

            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'DELL')]")))

            driver.find_element(
                By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'LG')]").click()

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_asus, input4)):
            print(input4)
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div._203eRC._2ssEMF")))

            driver.find_element(By.CSS_SELECTOR, "div._203eRC._2ssEMF").click()

            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'ASUS')]")))

            driver.find_element(
                By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'LG')]").click()

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._1fQZEK")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2
    """WATCHES"""
    if(re.search(pattern_smartwatches, input)):
        driver.get("https://www.flipkart.com")

        button_close = driver.find_element(By.CLASS_NAME, '_2doB4z')
        WebDriverWait(driver, 20)
        button_close.click()
        WebDriverWait(driver, 20)

        driver.find_element(
            By.CLASS_NAME, '_3704LK').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_apple, input4)):

            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a._2UzuFa")))

            links = driver.find_elements(By.CSS_SELECTOR, "a._2UzuFa")

            product_links = [link.get_attribute("href") for link in links]
            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "_396cs4 _2amPTt _3qGmMb"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_boat, input4)):
            print(input4)
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div._2gmUFU _3V8rao")))

            driver.find_element(By.CSS_SELECTOR, "div._2gmUFU _3V8rao").click()

            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'boAt')]")))

            driver.find_element(
                By.XPATH, "//div[contains(@class, '_3879cV') and contains(text(), 'boAt')]").click()

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.IRpwTa")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a.IRpwTa")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2

        if(re.search(pattern_firebolt, input4)):
            print(input4)
            WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div._203eRC._2ssEMF")))

            driver.find_element(By.CSS_SELECTOR, "div._203eRC._2ssEMF").click()

            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.IRpwTa")))

            product_links = []

            links = driver.find_elements(By.CSS_SELECTOR, "a.IRpwTa")

            for link in links:

                href = link.get_attribute("href")

                product_links.append(href)

            print(product_links)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in product_links]
            csv_data = csv_header + ''.join(csv_rows)
            with open('flipkart_products_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults2 = []
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
            produc_links = []
            with open('flipkart_products_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        product_page = requests.get(item, headers=headers)
                        print(product_page)
                        product_soup = BeautifulSoup(
                            product_page.content, 'html.parser')
                        try:
                            pname = product_soup.find(
                                'h1', {'class': 'yhB1nd'}).text.strip()
                            pprice = product_soup.find(
                                'div', {'class': '_30jeq3 _16Jk6d'}).text.strip()
                            product_image = product_soup.find(
                                'div', {"class": "CXW8mj"})
                            imageSrc = product_image.find('img')['src']
                            ratings = product_soup.find(
                                'div', {'class': '_3LWZlK'}).text.strip()
                            reviews = product_soup.select('.t-ZTKy')
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                                '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')
                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')
                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSrc
                            results["flipkart_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults2.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults2)

                return lresults2


def amazon_products(input):
    try:

        driver.get("https://www.amazon.in")
    finally:
        driver.get("https://www.amazon.in")

    if(re.search(pattern_tv, input)):
        WebDriverWait(driver, 10)

        driver.find_element(
            By.ID, 'twotabsearchtextbox').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_samsung, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:

                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')
                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)

                return lresults

        if(re.search(pattern_sony, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                    try:
                        pname = driver.find_element(
                            By.ID, "productTitle").text
                        pprice = driver.find_element(
                            By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                        productImage = driver.find_element(
                            By.CSS_SELECTOR, "#imgTagWrapperId img")
                        imageSource = productImage.get_attribute("src")
                        ratings = driver.find_element(
                            By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                        reviews = driver.find_elements(
                            By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                    except:
                        pname = "product name not found"
                        pprice = "price not found"
                        productImage = "product image not found"
                        imageSource = "product image not found"
                        ratings = "ratings not found"
                        reviews = "no reviews"
                    try:
                        product_rev = ""
                        pr = []
                        for revi in reviews:
                            revi_rev = revi.text
                            pr.append(revi_rev)

                        for i in pr:

                            loaded_rf = joblib.load("model.joblib")

                            test_text_preprocessed = [
                                preprocess_text(text) for text in [i]]

                            loaded_vectorizer = joblib.load(
                                'vectorizer.joblib')
                            test_text_vectorized = loaded_vectorizer.transform(
                                test_text_preprocessed)

                            predicted_ratings = loaded_rf.predict(
                                test_text_vectorized)
                            pr_list = []
                            pr_list.append(predicted_ratings)

                        '''linear regression'''
                        for i in pr:
                            loaded_rf2 = joblib.load("model2.joblib")

                            test_text_preprocessed = [
                                preprocess_text(text) for text in [i]]

                            loaded_vectorizer2 = joblib.load(
                                'vectorizer2.joblib')
                            test_text_vectorized2 = loaded_vectorizer2.transform(
                                test_text_preprocessed)

                            predicted_ratings2 = loaded_rf2.predict(
                                test_text_vectorized2)
                            pr_list2 = []
                            pr_list2.append(predicted_ratings2)

                        average2 = sum(pr_list2) / len(pr_list2)
                        av = abs(average2-1)*1

                        average_linear = np.array2string(
                            av, separator=', ')

                        average = sum(pr_list) / len(pr_list)
                        average_random = np.array2string(
                            average, separator=', ')

                        results = {}
                        results["product_name"] = pname
                        results['product_price'] = pprice
                        results["image_source"] = imageSource
                        results["amazon_ratings"] = ratings
                        results["random_forest"] = average_random
                        results["linearRegression"] = average_linear
                        results["product_link"] = item

                        lresults.append(results)

                        counter += 1
                    except:
                        print("no ratings found cannot predict")
            print(lresults)
            return lresults

        if(re.search(pattern_lg, input4)):

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults

        ''' mobiles'''

    if(re.search(pattern_mobile, input)):
        WebDriverWait(driver, 10)

        driver.find_element(
            By.ID, 'twotabsearchtextbox').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_samsung, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)

                return lresults

        if(re.search(pattern_apple, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults

        if(re.search(pattern_mi, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults
    ''' smartwatch'''

    if(re.search(pattern_smartwatches, input)):
        WebDriverWait(driver, 10)

        driver.find_element(
            By.ID, 'twotabsearchtextbox').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_boat, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)

                return lresults
        if(re.search(pattern_apple, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults
        if(re.search(pattern_noise, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults
    ''' laptop'''

    if(re.search(pattern_laptops, input)):
        WebDriverWait(driver, 10)

        driver.find_element(
            By.ID, 'twotabsearchtextbox').send_keys(input, Keys.RETURN)
        WebDriverWait(driver, 20)

        input4 = input.upper()
        print(input4)

        if(re.search(pattern_dell, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)

                return lresults
        if(re.search(pattern_asus, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults

        if(re.search(pattern_hp, input4)):
            WebDriverWait(driver, 20)

            links = driver.find_elements(
                By.XPATH, "//div//div//span//h2//a[contains(@class,'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal')]")
            productLinks = []
            for link in links:
                href = link.get_attribute("href")
                productLinks.append(href)

            csv_header = 'product_link\n'

            csv_rows = [f"{link}\n" for link in productLinks]
            csv_data = csv_header + ''.join(csv_rows)
            with open('amazon_product_links.csv', 'w') as file:
                file.write(csv_data)

            results_data = []
            res = []
            lresults = []
            with open('amazon_product_links.csv', 'r') as csvfile:

                reader = csv.reader(csvfile)

                next(reader)
                counter = 0

                for row in reader:
                    for item in row:
                        if counter >= 6:
                            break

                        time.sleep(15)
                        driver.get(item)

                        time.sleep(10)
                        try:
                            pname = driver.find_element(
                                By.ID, "productTitle").text
                            pprice = driver.find_element(
                                By.XPATH, "//div//div//span[contains(@class,'a-price aok-align-center reinventPricePriceToPayMargin priceToPay')]").text
                            productImage = driver.find_element(
                                By.CSS_SELECTOR, "#imgTagWrapperId img")
                            imageSource = productImage.get_attribute("src")
                            ratings = driver.find_element(
                                By.XPATH, "//div//div//div//div//span//span[contains(@data-hook, 'rating-out-of-text')]").text
                            reviews = driver.find_elements(
                                By.CSS_SELECTOR, ".a-expander-content.reviewText.review-text-content.a-expander-partial-collapse-content")
                        except:
                            pname = "product name not found"
                            pprice = "price not found"
                            productImage = "product image not found"
                            imageSource = "product image not found"
                            ratings = "ratings not found"
                            reviews = "no reviews"
                        try:
                            product_rev = ""
                            pr = []
                            for revi in reviews:
                                revi_rev = revi.text
                                pr.append(revi_rev)

                            for i in pr:

                                loaded_rf = joblib.load("model.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer = joblib.load(
                                    'vectorizer.joblib')
                                test_text_vectorized = loaded_vectorizer.transform(
                                    test_text_preprocessed)

                                predicted_ratings = loaded_rf.predict(
                                    test_text_vectorized)
                                pr_list = []
                                pr_list.append(predicted_ratings)

                            '''linear regression'''
                            for i in pr:
                                loaded_rf2 = joblib.load("model2.joblib")

                                test_text_preprocessed = [
                                    preprocess_text(text) for text in [i]]

                                loaded_vectorizer2 = joblib.load(
                                    'vectorizer2.joblib')
                                test_text_vectorized2 = loaded_vectorizer2.transform(
                                    test_text_preprocessed)

                                predicted_ratings2 = loaded_rf2.predict(
                                    test_text_vectorized2)
                                pr_list2 = []
                                pr_list2.append(predicted_ratings2)

                            average2 = sum(pr_list2) / len(pr_list2)
                            av = abs(average2-1)*1

                            average_linear = np.array2string(
                                av, separator=', ')

                            average = sum(pr_list) / len(pr_list)
                            average_random = np.array2string(
                                average, separator=', ')

                            results = {}
                            results["product_name"] = pname
                            results['product_price'] = pprice
                            results["image_source"] = imageSource
                            results["amazon_ratings"] = ratings
                            results["random_forest"] = average_random
                            results["linearRegression"] = average_linear
                            results["product_link"] = item

                            lresults.append(results)

                            counter += 1
                        except:
                            print("no ratings found cannot predict")
                print(lresults)
                return lresults


if __name__ == '__main__':
    app.run(port=5000, debug=True)
