import io
import json
import math
import os
import pickle
from random import shuffle
from time import sleep
from urllib.parse import urlencode
from uuid import uuid4

import PIL
import clip
import requests
import tensorflow
import torch
from PIL import Image
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# google, bing, unsplash, flickr, all
site = "all"
breed = "Labrador Retriever"

options = webdriver.ChromeOptions()
options.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
driver = webdriver.Chrome(options=options)
driver.maximize_window()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
classes = list(pickle.load(open("imagenet_classes.pkl", "rb")).values())
tokenized = list()

for c in classes:
    tokenized.append(clip.tokenize(c.strip().lower()))

if breed.lower() not in classes and breed not in classes:
    tokenized.append(clip.tokenize(breed.strip().lower()))
    classes.append(breed.strip().lower())


def download_images_from_urls(urls: list):
    train_path = "H:\\Datasets\\DogBreed\\train\\" + breed + "\\"
    val_path = "H:\\Datasets\\DogBreed\\val\\" + breed + "\\"
    test_path = "H:\\Datasets\\DogBreed\\test\\" + breed + "\\"
    train_amount = math.floor(len(urls) * 0.70)
    val_amount = math.floor(len(urls) * 0.15)
    dataset = "train"

    shuffle(urls)
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    if not os.path.exists(val_path):
        os.mkdir(val_path)

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    if not os.path.exists(train_path + "\\uncertain\\"):
        os.mkdir(train_path + "\\uncertain\\")

    if not os.path.exists(val_path + "\\uncertain\\"):
        os.mkdir(val_path + "\\uncertain\\")

    if not os.path.exists(test_path + "\\uncertain\\"):
        os.mkdir(test_path + "\\uncertain\\")

    img_num = 0
    for url in urls:
        if dataset == "train":
            if img_num >= train_amount:
                dataset = "val"

            save_path = train_path
        elif dataset == "val":
            if img_num >= train_amount + val_amount:
                dataset = "test"

            save_path = val_path
        else:
            save_path = test_path

        try:
            img_num += 1
            percentage = math.floor(img_num / len(urls) * 100)
            print("(" + str(percentage) + "%) Downloading image " + str(img_num) + " of " + str(len(urls)) + "...")
            random_uuid = uuid4().hex
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                response.raw.decode_content = True
                content = response.content
                image_file = io.BytesIO(content)
                image = Image.open(image_file).convert("RGB")
                image.thumbnail((600, 600), PIL.Image.BILINEAR)
                clip_image = preprocess(image).unsqueeze(0).to(device)
                text = torch.cat(tokenized).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(clip_image)
                    text_features = model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(1)

                for value, index in zip(values, indices):
                    print("Prediction: " + classes[index])

                    if breed.lower().strip() not in classes[index].lower().strip():
                        save_path += "\\uncertain\\"

                    file_path = save_path + random_uuid + ".jpg"
                    with open(file_path, "xb") as f:
                        image.save(f, "JPEG", quality=85)

                    tf_img = tensorflow.io.read_file(file_path)

                    try:
                        tensorflow.io.decode_image(tf_img)
                    except:
                        print("Invalid image! Removing...")
                        os.remove(file_path)
                        continue

                    if breed.lower().strip() not in classes[index]:
                        save_path.replace("\\uncertain\\", "")
        except Exception as ex:
            print(ex)
            print("Error! Skipping...")
            continue


def scrape_google():
    url = "https://www.google.com/search?" + urlencode({"q": breed + " dog", "tbm": "isch", "safe": "on"})
    minimum_images = 250

    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                    "/html/body/div[2]/c-wiz/div[4]/div[1]/div/div/div/div[1]/div[1]/span/div[1]/div[1]/div[1]/a[1]/div[1]/img")))
    image_box = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[4]")
    urls = list()

    image_num = 0
    while len(urls) < minimum_images:
        images = image_box.find_elements(By.TAG_NAME, "img")

        if image_num >= len(images):
            break

        image = images[image_num]
        image_num += 1

        try:
            if "CBvy7c" in image.get_attribute("class"):
                continue

            ActionChains(driver).move_to_element(image).click(image).perform()
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                            "/html/body/div[2]/c-wiz/div[4]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img")))

            img_elem = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[4]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img")
            url = img_elem.get_attribute("src")

            if "http" in url:
                if url in urls:
                    continue

                urls.append(url)

        except Exception as ex:
            print(ex)
            continue

    print("Found " + str(len(urls)) + " images!")

    return urls


def scrape_bing():
    url = "https://www.bing.com/images/search?" + urlencode({"q": breed + " dog", "first": 1, "tsc": "ImageBasicHover"})
    urls = list()
    image_num = 0
    minimum_images = 400

    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "iusc")))
    images = driver.find_elements(By.CLASS_NAME, "iusc")

    while len(urls) < minimum_images:
        if image_num >= len(images):
            try:
                driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
                WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CLASS_NAME, "loading")))
                sleep(1)
                images = driver.find_elements(By.CLASS_NAME, "iusc")

                if image_num >= len(images):
                    element = driver.find_element(By.CLASS_NAME, "btn_seemore")
                    driver.execute_script("arguments[0].click()", element)
                    WebDriverWait(driver, 10).until(EC.invisibility_of_element_located((By.CLASS_NAME, "loading")))
                    sleep(3)
                    images = driver.find_elements(By.CLASS_NAME, "iusc")

                    if image_num >= len(images):
                        break
            except:
                break
            continue

        image = images[image_num]
        image_num += 1

        try:
            m_attr = image.get_attribute("m")

            if m_attr is None:
                continue

            url = json.loads(m_attr)["murl"]

            if "http" in url:
                if url in urls:
                    continue

                urls.append(url)
        except Exception as ex:
            print(ex)
            images = driver.find_elements(By.CLASS_NAME, "iusc")
            continue

    print("Found " + str(len(urls)) + " images!")
    return urls


def scrape_unsplash():
    url = "https://unsplash.com/s/photos/" + breed.replace(" ", "-") + " dog"
    urls = list()
    image_num = 0
    minimum_images = 80

    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "img")))
    images = driver.find_elements(By.TAG_NAME, "img")

    while len(images) < minimum_images:
        if image_num >= len(images):
            break

        image = images[image_num]
        driver.execute_script("arguments[0].scrollIntoView();", image)
        sleep(0.2)
        images = driver.find_elements(By.TAG_NAME, "img")
        image_num += 1

    image_num = 0
    images = driver.find_elements(By.TAG_NAME, "img")

    while len(urls) < minimum_images:
        if image_num >= len(images):
            break

        image = images[image_num]
        image_num += 1

        try:
            if "https://images.unsplash.com/photo-" not in image.get_attribute("src"):
                continue

            url = image.get_attribute("src")

            if "http" in url:
                if url in urls:
                    continue

                urls.append(url)
        except Exception as ex:
            print(ex)
            print("Error! Skipping...")
            continue

    print("Found " + str(len(urls)) + " images!")
    return urls


def scrape_flickr():
    url = "https://www.flickr.com/search/?" + urlencode({"text": breed + " dog"})
    urls = list()
    image_num = 0
    minimum_images = 2000

    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "photo-list-photo-view")))
    sleep(5)
    images = driver.find_elements(By.CLASS_NAME, "photo-list-photo-view")

    while len(urls) < minimum_images:
        if image_num >= len(images):
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.CLASS_NAME, "flickr-dots")))
            images = driver.find_elements(By.CLASS_NAME, "photo-list-photo-view")

            if image_num >= len(images):
                try:
                    element = driver.find_element(By.CLASS_NAME, "infinite-scroll-load-more")
                    driver.execute_script("arguments[0].click()", element)
                    WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.CLASS_NAME, "flickr-dots")))
                    sleep(3)
                    images = driver.find_elements(By.CLASS_NAME, "photo-list-photo-view")

                    if image_num >= len(images):
                        print("Couldn't find more images!")
                        break
                except:
                    continue

        image = images[image_num]
        image_num += 1

        try:
            url = "https://" + image.get_attribute("style").split('background-image: url("//')[1].replace('");',
                                                                                                          "").replace(
                "_m.jpg", ".jpg").replace("_n.jpg", ".jpg")

            if "http" in url:
                if url in urls:
                    continue

                urls.append(url)
        except Exception as ex:
            print(ex)
            images = driver.find_elements(By.CLASS_NAME, "photo-list-photo-view")
            continue

    print("Found " + str(len(urls)) + " images!")
    return urls


url_list = list()

if site == "google":
    url_list.extend(scrape_google())
elif site == "bing":
    url_list.extend(scrape_bing())
elif site == "unsplash":
    url_list.extend(scrape_unsplash())
elif site == "flickr":
    url_list.extend(scrape_flickr())
elif site == "all":
    url_list.extend(scrape_flickr())
    url_list.extend(scrape_bing())
    url_list.extend(scrape_google())
    url_list.extend(scrape_unsplash())

driver.quit()
download_images_from_urls(url_list)
