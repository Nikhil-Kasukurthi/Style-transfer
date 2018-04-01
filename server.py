# Tornado Libraries
import tornado.ioloop
import tornado.web
from tornado.web import removeslash
import tornado.ioloop
import tornado.options
from tornado.gen import engine, Task, coroutine

from PIL import Image
import io
import numpy
import os
import sys
import time
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

import utils
from net import Net

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dataset_df = pd.read_csv('dataset.csv')
static_file_path = 'static'

model = './21styles.model'
style_images_path = 'images/museum_styles/'

style_model = Net(ngf=128)
style_model.load_state_dict(torch.load(model), False)
cuda = torch.cuda.is_available()

if cuda:
    style_model.cuda()
    content_image = content_image.cuda()
    style = style.cuda()


def evaluate(raw_content_image, raw_content_size, style_image, style_size, cuda, output_name):
    content_image = utils.tensor_load_rgbimage(
        raw_content_image, size=raw_content_size, keep_asp=True)
    content_image = content_image.unsqueeze(0)
    style = utils.tensor_load_rgbimage(style_image, size=style_size)
    style = style.unsqueeze(0)
    style = utils.preprocess_batch(style)

    style_v = Variable(style)

    content_image = Variable(utils.preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    transfer_image = utils.tensor_save_bgrimage(
        output.data[0], output_name, cuda)
    return transfer_image


class UploadHandler(tornado.web.RequestHandler):

    @removeslash
    @coroutine
    def post(self):
        start = time.time()
        file = self.request.files['file'][0]
        self.request.headers['Content-Type'] = 'multipart/form-data'
        # self.request.connection.set_max_body_size(100000000000000000000)
        style_id = self.get_argument('style_id')
        content = file['body']
        input_file = open("uploads/"
                           + file['filename'], 'wb')
        input_file.write(file['body'])
        input_file.close()
        print('Image upload', time.time() - start)
        start = time.time()
        style_image_path = style_images_path + style_id + '.jpg'
        image = "uploads/"+ file['filename']
        # image = Image.open("uploads/"+ file['filename']).convert('RGB')
        if os.path.exists(style_image_path):
            result_image = evaluate(
                image, 512,
                style_image_path,
                224, cuda,
                os.path.join(static_file_path, file['filename']))
            response = {}
            print('Response', time.time() - start)
            response['style_image'] = '/static/' + file['filename']
            self.write(response)
        else:
            self.write({'Response': 'Image not Found'})


class DatasetHandler(tornado.web.RequestHandler):

    def get(self):
        # print(dataset_df.head())
        dataset = {}
        dataset['images'] = []
        # print(dataset_df)
        for index, row in dataset_df.iterrows():
            item = {}
            item['Title'] = row['Title']
            item['Database ID'] = row['Database ID']
            item['Link'] = row['Link']
            item['Drive-Link'] = row['Drive-Link']
            dataset['images'].append(item)
        self.write(dataset)

app = tornado.web.Application([
    (r'/upload', UploadHandler),
    (r'/dataset', DatasetHandler)
], debug=True, static_path=static_file_path)

app.listen(9001)
tornado.ioloop.IOLoop.instance().start()
