import time
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS   
from PIL import Image
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, square
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage.filters import rank


class DataHandler(object):
    
    
    def __init__(self, bsObj):
        self.bsObj = bsObj
        
        self.msgs = bsObj.find_all('div', {'class':'entry unvoted'})
        self.thread_data = bsObj.find_all('p', {'class':'parent'})
        self.op_data = bsObj.find_all('div', {'data-context':'listing'})

        self.msg_list = []
        self.msg_time_list = []
        self.OP_list = []
        self.OP_time_list = []

    def get_int_from_score(self, string):
        return int(re.split('([0-9]*)', string)[1])
        
    def fill_data_dict(self, msg, thread_data):
        msg_dict = {'score': self.get_int_from_score(msg.find('span', {'class': 'score unvoted'}).get_text()),
                    'likes': self.get_int_from_score(msg.find('span', {'class': 'score likes'}).get_text()),
                    'dislikes': self.get_int_from_score(msg.find('span', {'class': 'score dislikes'}).get_text()),
                    'text': msg.find('form', {'action': '#'}).get_text(),
                    'thread_title': thread_data.find_all('a')[1].get_text(),
                    'thread_author': thread_data.find_all('a')[2].get_text(),
                    'subreddit': thread_data.find_all('a')[3].get_text(),
                    }   
        return msg_dict
        
        
    def fill_op_data_dict(self, OP):
        OP_dict = {'score': int(OP.find('div', {'class': 'score unvoted'}).get('title')),
                    'title': OP.find('a', {'data-event-action':'title'}).get_text(),
                    'domain': OP.find('span', {'class':'domain'}).get_text(),
                    'OP': OP.find('p', {'class': 'tagline'}).find_all('a')[0].get_text(),
                    'subreddit': OP.find('p', {'class': 'tagline'}).find_all('a')[1].get_text()
                    }
        return OP_dict    
    
    def get_msg_data(self, bsObj):
        for i, (msg, thread_data) in enumerate(zip(self.msgs, self.thread_data)):
            print "msg: {}/{}".format(i+1, len(self.msgs))
            if len(thread_data.get_text()) > 0:
                self.msg_time_list.append(pd.to_datetime(msg.find('time').get('datetime')))
                msg_dict = self.fill_data_dict(msg, thread_data)
                self.msg_list.append(msg_dict)
        return pd.DataFrame(self.msg_list, 
                            index = self.msg_time_list)
    
    def get_OP_data(self, bsObj):
        for i, OP in enumerate(self.op_data):
            print "OP: {}/{}".format(i+1, len(self.op_data))
            self.OP_time_list.append(pd.to_datetime(OP.find('time').get('datetime')))
            self.OP_list.append(self.fill_op_data_dict(OP))
        return pd.DataFrame(self.OP_list, 
                            index = self.OP_time_list)
                
    def get_data(self):
        msg_data = self.get_msg_data(self.bsObj)
        OP_data = self.get_OP_data(self.bsObj)
        try:
            next_page_url =  self.bsObj.find('span', {'class': 'next-button'}).find('a')['href']
        except AttributeError:
            next_page_url = None
        return msg_data, OP_data, next_page_url

    
    
class GetAccData(object):
    
    
    def __init__(self, url):
        self.seed_url = url
        self.headers = {'user-agent': 'gratis_poepjes'}
        self.comment_list = []
        self.OP_list = []


    def update_data_list(self, msg_data, OP_data):
        self.comment_list.append(msg_data)
        self.OP_list.append(OP_data)      
        
    def make_request(self, url):
        p1 = requests.get(url, headers = self.headers)
        print url, p1.status_code
        html_content = p1.content
        body = BeautifulSoup(html_content, 'lxml')
        msg_data, OP_data, next_page_url = DataHandler(body).get_data()
        self.update_data_list(msg_data, OP_data)   
        return next_page_url

    def while_loop(self):
        next_page_url = self.make_request(self.seed_url)
        counter = 0
        while next_page_url is not None:
            print "DOWNLOAD COUNTER: {}".format(counter)
            time.sleep(4)
            counter += 1
            next_page_url = self.make_request(next_page_url)
        return True
        
    def work(self):
        self.while_loop()
        OP_df = pd.concat(self.OP_list)
        comment_df = pd.concat(self.comment_list)
        return OP_df, comment_df
        
        
class GetImgMask(object):
    
    def apply_otsu(self, img, square_size):
        thresh = threshold_otsu(img)
        return closing(img < thresh, square(square_size))
        
    def get_label_image_with_otsu(self, imgray):
        bw = self.apply_otsu(imgray, square_size = 15)
        label_image = label(bw)
        borders = np.logical_xor(bw, bw)
        label_image[borders] = -1
        return bw, label_image    
        
    def get_mask(self, img_path):
        imgray = io.imread(imgpath, as_grey = True)
        bw, label_image = self.get_label_image_with_otsu(imgray)
        
        denoised = rank.median(bw, disk(2))
        markers = rank.gradient(denoised, disk(21)) < 10
        markers = ndi.label(markers)[0]
    
        gradient = rank.gradient(denoised, disk(5))
        
        labels = watershed(gradient, markers)
        extracted_square = labels == 54  #Manually extract the label value
        filled_image = ndi.binary_fill_holes(extracted_square)
        return filled_image.astype('uint8') * 255    


class AccStats(object):
    
    
    def __init__(self, OP_df, comment_df):
        self.OP_df = OP_df
        self.comment_df = comment_df
        self.title_fontsize = 18
        self.tick_fontsize = 15
        self.label_fontsize = 10
        
    def split_all_words(self, comment_thread):
        word_list, split_list = [], []
        for post in comment_thread.text:
            split_list.append(post.split())
            for word in post.split():
                word_list.append(word)
        return word_list, split_list
        
    def calc_der(self, val_list):
        return [val_list[i+1] - val_list[i] for i in range(len(val_list) -1)] 

    def get_color_tuple(self, val, upperbound):
        color_val = (val/ float(upperbound)) 
        return (1.0 - color_val, 0, color_val)
        
    def merge_word_list(self, word_list):
        string =  ''
        for word in word_list:
            string += ' ' + word
        return string

    def get_data_from_thread(self, thread_title):
        self.thread_OP = self.OP_df.loc[self.OP_df.title == thread_title]
        self.comment_thread = self.comment_df.loc[self.comment_df.thread_title == thread_title]
        self.comment_second_list = self.comment_thread.index.values.astype('timedelta64[s]').astype('uint64') / (1000000000)
        self.time_between_posts = self.calc_der(sorted(self.comment_second_list))
        
        self.word_list, self.split_list = self.split_all_words(self.comment_thread)
        self.word_length_per_post = map(len, self.split_list) #Plot?
        self.word_length_per_word = map(len, self.word_list)

        
            
    def print_stats(self):
        print "Total amount of posts in AMA: {}".format(len(self.comment_thread))
        print "Total amount of time spend responding: {}".format(self.comment_thread.index[0] - self.comment_thread.index[-1])
        print "Amount of post / thread / link karma gained: {}".format(self.thread_OP.score.values[0])
        print "Total amount of comment karma gained: {}".format(sum(self.comment_thread.score))
        print "Comment karma gained: Min {} Max {} Average {} Median {}".format(min(self.comment_thread.score), 
                                                                                max(self.comment_thread.score),
                                                                                int(np.average(self.comment_thread.score)),
                                                                                int(np.median(self.comment_thread.score)))      
        print "Total amount of words: {}".format(len(self.word_list))
        print "Amount of words per post: Min {} Max {} Average {} Median {}".format(min(self.word_length_per_post), 
                                                                                max(self.word_length_per_post),
                                                                                int(np.average(self.word_length_per_post)),
                                                                                int(np.median(self.word_length_per_post)))
        
        
        
        print "Time between posts (s): Min {} Max {} Average {} Median {}".format(min(self.time_between_posts), 
                                                                                max(self.time_between_posts),
                                                                                int(np.average(self.time_between_posts)),
                                                                                int(np.median(self.time_between_posts)))
        
        
        print "Word length: Min {} Max {} Average {} Median {}".format(min(self.word_length_per_word), 
                                                                                max(self.word_length_per_word),
                                                                                int(np.average(self.word_length_per_word)),
                                                                                int(np.median(self.word_length_per_word)))
         
        
        print "Comment karma gained: Min {} Max {} Average {} Median {}".format(min(self.comment_thread.score), 
                                                                                max(self.comment_thread.score),
                                                                                int(np.average(self.comment_thread.score)),
                                                                                int(np.median(self.comment_thread.score)))
    def make_wordcloud(self, mask = None):
        if mask is not None:
            mask = abs(mask - 1)
            wordcloud = WordCloud(stopwords = STOPWORDS, mask = mask)        
        else:
            wordcloud = WordCloud(stopwords = STOPWORDS)
            
        string = self.merge_word_list(self.word_list)
        wc = wordcloud.generate(string)
        plt.figure(figsize = (16, 10))
        plt.imshow(wc)
        plt.show()
        return wc
        
        
    def get_wordcloud_in_original_image(self, imgpath, maskpath, savepath):
        img = np.array(Image.open(imgpath))
        mask = np.array(Image.open(maskpath))
        
        wc_img = self.make_wordcloud(mask)
        wc_img_ar = np.array(wc_img)
        mask_coords = np.where(mask > 1)
        
        for i, j in zip(mask_coords[0], mask_coords[1]):
            img[i, j, :] = wc_img_ar[i, j, :]   

        io.imsave(savepath, img)
        plt.figure(figsize = (16, 10))
        plt.imshow(img)    
        plt.show()
        
    def generate_hist_plots(self):
        gs = gridspec.GridSpec(2, 2, height_ratios = [1, 1])
        plt.figure(figsize = (16, 10))
        
        ax1 = plt.subplot(gs[0, 0])
        
        ax1.hist(self.word_length_per_post)
        plt.title('Amount of words per post', fontsize = self.title_fontsize)
        plt.xlabel('Amount of words', fontsize = self.label_fontsize)
        plt.ylabel('Occurence count', fontsize = self.label_fontsize)
            
        
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(self.comment_thread.score, bins = 50)
        plt.title('Amount of karma per post', fontsize = self.title_fontsize)
        plt.xlabel('Amount of karma', fontsize = self.label_fontsize)
        plt.ylabel('Occurence count', fontsize = self.label_fontsize)
        
        ax3 = plt.subplot(gs[0, 1])
        ax3.hist(self.word_length_per_word)
        plt.title('Word length', fontsize = self.title_fontsize)
        plt.xlabel('Word length', fontsize = self.label_fontsize)
        plt.ylabel('Occurence count', fontsize = self.label_fontsize)
            
        ax4 = plt.subplot(gs[1, 1])
        ax4.hist(self.time_between_posts, bins = 30)
        plt.title('Time between posts', fontsize = self.title_fontsize)
        plt.xlabel('Amount of seconds between each post', fontsize = self.label_fontsize)
        plt.ylabel('Occurence count', fontsize = self.label_fontsize)
        plt.show()       
   
    def generate_fancy_time_plot(self):
 
        plt.figure(figsize = (16, 10))
        gs = gridspec.GridSpec(1, 10, height_ratios = [6, 1])
        ax1 = plt.subplot(gs[0, :9])
          
        end = self.comment_second_list[0]
        start = self.comment_second_list[-1]
        step_size = 180
        tick_end = (int(end - start) / step_size) + 1
        location_vals = start + range(0, step_size * (tick_end + 1), step_size)
        label_vals = [str(pd.to_datetime(val, unit = 's'))[-8:] for val in location_vals]
        
                      
        second_hist = np.histogram(self.comment_second_list, bins = 200)
        comment_length = self.comment_thread.text.apply(len).values
        lower_bound, upper_bound = min(comment_length), max(comment_length)
        
        counter = 0
        for bin_nr, val in zip(second_hist[1], second_hist[0]):
            if val == 1:
                col = self.get_color_tuple(comment_length[counter], upper_bound)
                ax1.bar(bin_nr, val, color = col, edgecolor = col)
                counter += 1
        
        plt.title('Amount of posts on given time interval. One bar = One post. Total time: {}'.format(self.comment_thread.index[0] - self.comment_thread.index[-1]) , 
                  fontsize = self.title_fontsize)
        plt.xlabel('Time from first post to last post', fontsize = self.label_fontsize)
        plt.xticks(location_vals, label_vals, rotation = '-30', fontsize = self.tick_fontsize)
        plt.yticks([])
        
        
        #Make gradient
        width = 20
        gradient = np.zeros(((upper_bound - lower_bound) * 2, width, 3))
        for i in range(lower_bound, upper_bound):
            color = self.get_color_tuple(i, upper_bound)
            gradient[(i - lower_bound)*2:((i - lower_bound)*2 + 2), :, :] = np.ones((2, width, 3)) * color
        
                     
        location_vals = range(0, (upper_bound - lower_bound) * 2, 30) 
        label_vals = range(lower_bound, upper_bound, (upper_bound - lower_bound) / len(location_vals))
          
        ax2 = plt.subplot(gs[0, 9])
        ax2.imshow(gradient)
        plt.yticks(location_vals, label_vals, fontsize = self.tick_fontsize)
        plt.ylabel('Amount of characters per post', fontsize = self.label_fontsize)
        plt.xticks([])
        plt.tick_params(left='off', right='on')
        plt.show()

        

start = time.time()
url = 'https://www.reddit.com/user/EwanMcGregorT2'
#Dl data
dataObj = GetAccData(url)
OP_df, comment_df = dataObj.work()


        
thread_title = u"I'm Ewan McGregor, star of T2 TRAINSPOTTING - AMA!"

path = 'C:/'
imgpath = path + 'Nash4B2.jpg'
maskpath = path + 'mask.png'
savepath = path + 'wc_img.png'

img_mask = GetImgMask().get_mask(imgpath)
io.imsave(maskpath, img_mask)

AccObj = AccStats(OP_df, comment_df)
AccObj.get_data_from_thread(thread_title)
AccObj.print_stats()
AccObj.make_wordcloud()
AccObj.get_wordcloud_in_original_image(imgpath, maskpath, savepath)
AccObj.generate_hist_plots()
AccObj.generate_fancy_time_plot()

print "DONE: {}".format(time.time() - start)

