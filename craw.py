import time
import csv

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import FirefoxOptions

def print_red(text):
    print("\033[91m{}\033[00m".format(text))

def StrToFloat(num_str):
    if 'K' in num_str:
        if len(num_str) > 1:
            return float(num_str.replace('K', '')) * 1000
        return 1000.0
    if 'M' in num_str:
        if len(num_str) > 1:
            return float(num_str.replace('M', '')) * 1000000
        return 1000000.0
    if 'B' in num_str:
        if len(num_str) > 1:
            return float(num_str.replace('B', '')) * 1000000000
        return 1000000000.0

    return float(num_str)

def processValue(num_str):
    if 'views' in num_str:
        num_str = num_str[:-5]
    if 'LIKE' in num_str or 'DISLIKE' in num_str:
        return 0.0

    return StrToFloat(num_str)


class YoutubeCrawler:
    def __init__(self, channels, driver_path, csv_name = "youtube_data.csv"):
        self.channels = channels
        self.driver_path = driver_path
        self.csv_name = csv_name
        self._setCSVWriter(self.csv_name)

    def _initWebDriver(self):
        print("Start setting web driver ...")
        print("Driver path = {} ...    ".format(self.driver_path), end = '')
        
        try:
            opts = FirefoxOptions()
            opts.add_argument("--headless")
            profile = webdriver.FirefoxProfile()
            profile.set_preference("browser.cache.disk.enable", False)
            profile.set_preference("browser.cache.memory.enable", False)
            profile.set_preference("browser.cache.offline.enable", False)
            profile.set_preference("network.http.use-cache", False)
            self.driver = webdriver.Firefox(executable_path = self.driver_path, firefox_options = opts,
                                            firefox_profile = profile)
        except:
            print_red("Set web driver fail !!!")
            exit(-1)

        print("Done !!!")

    def _closeWebDriver(self):
        print("Close web driver ...    ")   

        try:
            self.driver.quit()
        except:
            print_red("Close web driver fail !!!")
            exit(-1)
            
        print("Done !!!")

    def _setCSVWriter(self, csv_name):
        print("Start initing csv file ... ")
        print("CSV file = {} ...    ".format(csv_name), end = '')

        try:
            self.csv_name = csv_name
            self.fieldnames = ['Channel', 'Title', 'View', 'Like', 'Dislike', 'Link']
            with open(self.csv_name, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()
        except:
            print_red("Init csv file fail !!!")
            exit(-1)

        print("Done !!!")

    def _writeCSV(self, channel, title, view, like, dislike, link):
        try:
            with open(self.csv_name, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writerow({'Channel': channel, 'Title': title, 'View': view, 
                                 'Like': like, 'Dislike': dislike, 'Link': link})
        except:
            raise Exception()

    def _getTitleView(self, channel):
        self._initWebDriver()

        try:
            self.driver.get(channel)
            time.sleep(3)
        except:
            print_red("Connect to channel page fail !!!")
            return

        try:
            ht = self.driver.execute_script("return document.documentElement.scrollHeight;")
            while True:
                prev_ht = self.driver.execute_script("return document.documentElement.scrollHeight;")
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
                ht = self.driver.execute_script("return document.documentElement.scrollHeight;")
                if prev_ht == ht:
                    break
        except:
            print_red("Scroll page fail !!!")
            return

        titles = []
        views = [] 
        links = []

        try:
            channel_name = self.driver.find_elements_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-browse/div[3]/ytd-c4-tabbed-header-renderer/app-header-layout/div/app-header/div[2]/div[2]/div/div[1]/div/div[1]/ytd-channel-name/div/div/yt-formatted-string')
            channel_name = channel_name[0].text
            print("Channel name = {}".format(channel_name))

            for title in self.driver.find_elements_by_xpath('//div/h3/a'):
                titles.append(title.text)

            for view in self.driver.find_elements_by_xpath('//*[@id="metadata-line"]/span[1]'):
                views.append(processValue(view.text))

            for video in self.driver.find_elements_by_xpath('//*[@id="video-title"]'):
                link = video.get_attribute('href')
                links.append(link)
        except:
            print_red("Get information from channel page fail !!!")
            return

        for idx, (title, view, link) in enumerate(zip(titles, views, links)):
            if idx % 50 == 49:
                self._closeWebDriver()
                self._initWebDriver()

            try:
                self.driver.get(link)
                time.sleep(3)
            except:
                print_red("Connect to video page fail !!!")
                return

            try:
                like_elm = self.driver.find_elements_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[5]/div[2]/ytd-video-primary-info-renderer/div/div/div[3]/div/ytd-menu-renderer/div/ytd-toggle-button-renderer[1]/a/yt-formatted-string')
                dislike_elm = self.driver.find_elements_by_xpath('/html/body/ytd-app/div/ytd-page-manager/ytd-watch-flexy/div[4]/div[1]/div/div[5]/div[2]/ytd-video-primary-info-renderer/div/div/div[3]/div/ytd-menu-renderer/div/ytd-toggle-button-renderer[2]/a/yt-formatted-string')
                like = like_elm[0].text
                dislike = dislike_elm[0].text
            except:
                print_red("Get information from video page fail !!!")
                return
            
            try:
                self._writeCSV(channel_name, title, view, like, dislike, link)
            except:
                print_red("Write data to csv file fail !!!")
                return
                
        self._closeWebDriver()

    def run(self):
        print("Start crawling youtube data ... ")
        
        for channel in self.channels:
            print("Youtube channel = {} ...    ".format(channel))
            self._getTitleView(channel)
            print("Done !!!")

        print("Finish crawling youbube data !!!")

def main():
    # Youtube channels (in videos page)
    channel_list = ["https://www.youtube.com/c/itsgrace/videos", "https://www.youtube.com/c/joerogan/videos", "https://www.youtube.com/c/roosterteeth/videos", "https://www.youtube.com/c/NoJumper/videos", "https://www.youtube.com/c/JREClips/videos", "https://www.youtube.com/c/TrueGeordie/videos", "https://www.youtube.com/c/H3Podcast/videos", "https://www.youtube.com/user/JennaJulienPodcast/videos", "https://www.youtube.com/c/JoeyDiaz/videos"]
    crawler = YoutubeCrawler(channel_list, 
                            "/home/redoao/Desktop/data_proj/driver/geckodriver", 
                            "podcast.csv")
    crawler.run()

if __name__ == '__main__':
    main()