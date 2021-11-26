import base64
import json
import os
import time
import urllib.parse

import cv2
import numpy
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from cube.cfop.CFOPsolver import CubeSolver
from cube.parse import graph, dataParse
from .tools import getResults

tempSolve = ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
color_result = {'O': '#', 'B': '#', 'R': '#', 'Y': '#', 'W': '#', 'G': '#'}
state_tesult = []


def index(request):
    return render(request, 'cube/index.html')
    # return HttpResponse("Hello, world. You're at the cube index.")


def test(request):
    return render(request, 'cube/test.html')


def basic(request):
    return render(request, 'cube/basic.html')


@csrf_exempt
def basic_initState(request):
    if request.method == 'POST':
        if request.is_ajax():
            with open('cube/static/json/initState.json', 'r') as f:
                result = json.load(f)
            return HttpResponse(json.dumps(result))
    return render(request, 'cube/basic.html')


@csrf_exempt
def basic_solve(request):
    if request.method == 'POST':
        if request.is_ajax():
            data = request.body.decode()
            data = urllib.parse.unquote(data)
            # print(data)
            data = data[7:-1].split(',')
            # print("data:", data)
            state = []
            for i in data:
                state.append(int(i))
            print("input state:", state)
            start = time.time()
            result = CubeSolver.getResults(state)
            print("complete!", "time use :", time.time() - start)
            print("result form:", result)
            return HttpResponse(json.dumps(result))
        else:
            print('null data')
    return render(request, 'cube/basic.html')


def advance(request):
    return render(request, 'cube/advance.html')


@csrf_exempt
def initState(request):
    if request.method == 'POST':
        if request.is_ajax():
            with open('cube/static/json/initState.json', 'r') as f:
                result = json.load(f)
            return HttpResponse(json.dumps(result))
    return render(request, 'cube/advance.html')


@csrf_exempt
def solve(request):
    if request.method == 'POST':
        if request.is_ajax():
            data = request.body.decode()
            data = urllib.parse.unquote(data)
            print(data)
            data = data[7:-1].split(',')
            # print("data:", data)
            state = []
            for i in data:
                state.append(int(i))
            print("input state:", state)
            start = time.time()
            result = getResults(state)
            print("complete!", "time use :", time.time() - start)
            print("result init form:", result)
            result['robot_solve_text'] = ''
            robot = []
            for i, v in enumerate(result['solve_text']):
                if "'" in v:
                    robot.append(v[0].lower())
                else:
                    robot.append(v)
            result['robot_solve_text'] = robot
            print("result robot form:", result)
            return HttpResponse(json.dumps(result))
        else:
            print('null data')
    return render(request, 'cube/advance.html')


@csrf_exempt
def upload(request):
    if request.method == 'POST':
        if request.is_ajax():
            data = request.body.decode('utf-8')
            json_data = json.loads(data)
            str_image = json_data.get("imgData")
            # print(str_image)
            img = base64.b64decode(str_image)
            img_np = numpy.fromstring(img, dtype='uint8')
            new_img_np = cv2.imdecode(img_np, 1)
            # if not os.path.exists('images'):
            #     os.mkdir('images')
            # name = time.strftime('%Y%m%d%H%M%S') + '.jpg'
            name = str(json_data.get("id")) + '.jpg'
            img_path = os.path.join('E:/pycharm/SuperCubeRobot/cube/images', name)
            cv2.imwrite(img_path, new_img_np)
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            tempSolve = graph.solve(hsv)
            color_result[tempSolve[4]] = tempSolve
            print(color_result)
            print('data:{}'.format(tempSolve))
            res = dict()
            res['color'] = tempSolve
            return HttpResponse(json.dumps(res))
    return render(request, 'cube/upload.html')


@csrf_exempt
def robot_solve(request):
    if request.method == 'POST':
        if request.is_ajax():
            data = request.body.decode()
            data = urllib.parse.unquote(data)
            # print(data)
            json_data = json.loads(data[7:])
            # print(json_data)
            # colors = json_data.get('colors')
            # print(colors)
            state = dataParse.parse(json_data)
            print(state)
            # state = json.loads(colors_data)
            # data = data[7:-1].split(',')
            # # print("data:", data)
            # state = []
            # for i in data:
            #     state.append(int(i))
            # print("input state:", state)
            start = time.time()
            result = getResults(state)
            print("complete!", "time use :", time.time() - start)
            print("result init form:", result)
            result['robot_solve_text'] = ''
            robot = []
            for i, v in enumerate(result['solve_text']):
                if "'" in v:
                    robot.append(v[0].lower())
                else:
                    robot.append(v)
            result['robot_solve_text'] = robot
            print("result robot form:", result)
            return HttpResponse(json.dumps(result))
        else:
            return HttpResponse(json.dumps({'error_message': 'error state'}))
    # return render(request, 'cube/advance.html')
