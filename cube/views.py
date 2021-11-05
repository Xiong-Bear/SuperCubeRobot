import base64
import os
import time

import numpy
from io import BytesIO
from PIL import Image
import re
import base64

from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt

from SuperCubeRobot import settings
from django.shortcuts import render
from django.http import HttpResponse
from .models import *
import json
import cv2


def index(request):
    return render(request, 'cube/index.html')
    # return HttpResponse("Hello, world. You're at the cube index.")


def test(request):
    return render(request, 'cube/test.html')


# def upload(request):
#     if request.method == 'POST':
#         if request.is_ajax():
#             image = request.FILES.get('image')
#             if not os.path.exists('images'):
#                 os.mkdir('images')
#             image.name = time.strftime('%Y%m%d%H%M%S') + '.jpg'
#             img_path = os.path.join('images', image.name)
#             # Start writing to the disk
#             with open(img_path, 'wb+') as destination:
#                 if image.multiple_chunks:  # size is over than 2.5 Mb
#                     for chunk in image.chunks():
#                         destination.write(chunk)
#                 else:
#                     destination.write(image.read())
#                 data = {'url': '../images/{}'.format(image.name)}
#             return HttpResponse(json.dumps(data))
#     return render(request, 'cube/upload.html')

@csrf_exempt
def upload(request):
    if request.method == 'POST':
        if request.is_ajax():
            data = request.body.decode('utf-8')
            json_data = json.loads(data)
            str_image = json_data.get("imgData")
            img = base64.b64decode(str_image)
            img_np = numpy.fromstring(img, dtype='uint8')
            new_img_np = cv2.imdecode(img_np, 1)
            if not os.path.exists('images'):
                os.mkdir('images')
            name = time.strftime('%Y%m%d%H%M%S') + '.jpg'
            img_path = os.path.join('images', name)
            cv2.imwrite(img_path, new_img_np)
            print('data:{}'.format('success'))
            return HttpResponse(json.dumps(data))
    return render(request, 'cube/upload.html')
