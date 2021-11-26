# coding=utf-8
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont




def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def judgeColor(hsv):
    h,s,v=hsv[0],hsv[1],hsv[2]
    if h>=24 and h<=45 and s>=43 and s<=255 and v>=46 and v<=255:
        return 'Y'
    elif h>=0 and h<=180 and s>=0 and s<=105 and v>=155 and v<=255:
        return 'W'
    elif h>=140 and h<=180 and s>=43 and s<=255 and v>=46 and v<=255:
        return 'R'
    # elif h>=0 and h<=10 and s>=43 and s<=255 and v>=46 and v<=255:
    #     return 'R'
    elif h>=45 and h<=86 and s>=43 and s<=255 and v>=46 and v<=255:
        return 'G'
    #elif h>=170 and h<=180 and s>=43 and s<=255 and v>=46 and v<=255:
        #return 'O'
    elif h>=0 and h<=20 and s>=43 and s<=255 and v>=46 and v<=255:
        return 'O'
    elif h>=100 and h<=124 and s>=43 and s<=255 and v>=46 and v<=255:
        return 'B'


def findMost(y,x,hsv):
    tempData={'B':0,'G':0,'W':0,'Y':0,'O':0,'R':0}
    for i in range(-5,6):
        for j in range(-5,6):
            #print(str(x+i)+'---'+str(y+i))
            #print(judgeColor(hsv[x+i,y+i]))
            t = judgeColor(hsv[x+i,y+i])
            if t in tempData.keys():
                tempData[t]+=1
    flag = -1
    k = ''
    for key, value in tempData.items():
        #print(tempData.get(key))
        if value > flag:
            flag = value
            k = key
    return k

def solve(hsv):
    result=[]
    result.append(findMost(225,145,hsv))
    result.append(findMost(295,145,hsv))
    result.append(findMost(365,145,hsv))
    result.append(findMost(225,215,hsv))
    result.append(findMost(295,215,hsv))
    result.append(findMost(365,215,hsv))
    result.append(findMost(225,285,hsv))
    result.append(findMost(295,285,hsv))
    result.append(findMost(365,285,hsv))
    return result

def check_dic(n):
    for v in n.values():
        if v == '#':
            return False
    return True

def colorToRGB(color):
    if color=='W':
        return [255,255,255]
    elif color=='B':
        return [0,0,255]
    elif color=='G':
        return [0,205,0]
    elif color=='Y':
        return [255,255,0]
    elif color=='R':
        return [205,0,0]
    elif color=='O':
        return [255,165,0]
    else:
        return [0,0,0]

def tips(c,frame):
    if c==0:
        return cv2ImgAddText(frame,'A:绿色块中心向前,白色中心块向上',160, 50, (55,255,155), 20)
    elif c==1:
        return cv2ImgAddText(frame,'B:红色块中心向前,白色中心块向上',160, 50, (55,255,155), 20)
    elif c==2:
        return cv2ImgAddText(frame,'C:橙色块中心向前,白色中心块向上',160, 50, (55,255,155), 20)
    elif c==3:
        return cv2ImgAddText(frame,'D:蓝色块中心向前,黄色中心块向上',160, 50, (55,255,155), 20)
    elif c==4:
        return cv2ImgAddText(frame,'E:黄色块中心向前,绿色中心块向上',160, 50, (55,255,155), 20)
    elif c==5:
        return cv2ImgAddText(frame,'F:白色块中心向前,蓝色中心块向上',160, 50, (55,255,155), 20)

def tips2(state,frame):
    if state==0:
        return cv2ImgAddText(frame,'按s捕获,q退出',160,340, (55,255,155), 20)
    elif state==1:
        return cv2ImgAddText(frame,'若结果正确按y,错误继续s捕获',160,340, (55,255,155), 20)
    elif state==2:
        return cv2ImgAddText(frame,'存储成功,继续s捕获',160,340, (55,255,155), 20)
    elif state==3:
        return cv2ImgAddText(frame,'该面已经拍照,无需重复拍照',160,340, (55,255,155), 20)
    elif state==4:
        return cv2ImgAddText(frame,'上次捕获有错误,重新按s捕获',160,340, (55,255,155), 20)
def showTem(tempSolve,frame):
    l=50
    location=[]
    RGB=colorToRGB(tempSolve[0])
    cv2.rectangle(frame, (461,101), (509,149), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[1])
    cv2.rectangle(frame, (461+l,101), (509+l,149), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[2])
    cv2.rectangle(frame, (461+2*l,101), (509+2*l,149), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[3])
    cv2.rectangle(frame, (461,101+l), (509,149+l), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[4])
    cv2.rectangle(frame, (461+l,101+l), (509+l,149+l), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[5])
    cv2.rectangle(frame, (461+2*l,101+l), (509+2*l,149+l), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[6])
    cv2.rectangle(frame, (461,101+2*l), (509,149+2*l), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[7])
    cv2.rectangle(frame, (461+l,101+2*l), (509+l,149+2*l), (RGB[2],RGB[1],RGB[0]), -1)
    RGB=colorToRGB(tempSolve[8])
    cv2.rectangle(frame, (461+2*l,101+2*l), (509+2*l,149+2*l), (RGB[2],RGB[1],RGB[0]), -1)

def showTem2(tempList,frame):
    l=30
    location=[[46,101,74,129],[46,221,74,249],[46,341,74,369],[440,101,468,129],[440,221,468,249],[440,341,468,369]]
    x=[0,1,2,0,1,2,0,1,2]
    y=[0,0,0,1,1,1,2,2,2]
    for i in location:
        for j in range(9):
            RGB=colorToRGB(tempList[location.index(i)][j])
            cv2.rectangle(frame, (i[0]+l*x[j],i[1]+l*y[j]), (i[2]+l*x[j],i[3]+l*y[j]), (RGB[2],RGB[1],RGB[0]), -1)

def findLocation(xx,yy):
    l=30
    location=[[46,101,74,129],[46,221,74,249],[46,341,74,369],[440,101,468,129],[440,221,468,249],[440,341,468,369]]
    x=[0,1,2,0,1,2,0,1,2]
    y=[0,0,0,1,1,1,2,2,2]
    for i in location:
        for j in range(9):
            if i[0]+l*x[j]<=xx and i[2]+l*x[j]>= xx and i[1]+l*y[j]<=yy and i[3]+l*y[j]>=yy:
                return [location.index(i),j]
    return[-1,-1]

def nextColor(color):
    c=['G','B','W','Y','R','O']
    if color=='O':
        return 'G'
    elif color=='X':
        return 'G'
    else:
        return c[c.index(color)+1]
        
def captureGraph(w):
    fflag=True
    result = {'O':'#','B':'#','R':'#','Y':'#','W':'#','G':'#'}
    #获取摄像头视频
    cap = cv2.VideoCapture(0)
    # 获取视频宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #frame_width = 100
    # 获取视频高度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_height = 100
    #print(frame_height)
    long=70
    state=0
    num=0
    tempSolve=['X','X','X','X','X','X','X','X','X']
    tempList=[['X','X','X','X','X','X','X','X','X'] for i in range(6)]
    def mouse_click(event, x, y, flags, para):
        nonlocal tempList
        nonlocal tempSolve
        if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
            #print('PIX:', x, y)
            location=findLocation(x,y)
            if not location[0]==-1:
                #print(location)
                tempList[location[0]][location[1]]=nextColor(tempList[location[0]][location[1]])
                tempSolve[location[1]]=tempList[location[0]][location[1]]
                #print("BGR:", img[y, x])
                #print("GRAY:", gray[y, x])
                #print("HSV:", hsv[y, x])
                #print(judgeColor(hsv[y, x]))
    c=0
    if w==0:
        state=0
    elif w==1:
        state=4
    print('输入s进行捕获，输入q退出')
    while (cap.isOpened()):
        ret,frame = cap.read()  
        cv2.rectangle(frame, (180,100), (180+long,100+long), (0,255,0), 2)
        cv2.rectangle(frame, (180+long,100), (180+2*long,100+long), (0,255,0), 2)
        cv2.rectangle(frame, (180+2*long,100), (180+3*long,100+long), (0,255,0), 2)
        cv2.rectangle(frame, (180,100+long), (180+long,100+2*long), (0,255,0), 2)
        cv2.rectangle(frame, (180+long,100+long), (180+2*long,100+2*long), (0,255,0), 2)
        cv2.rectangle(frame, (180+2*long,100+long), (180+3*long,100+2*long), (0,255,0), 2)
        cv2.rectangle(frame, (180,100+2*long), (180+long,100+3*long), (0,255,0), 2)
        cv2.rectangle(frame, (180+long,100+2*long), (180+2*long,100+3*long), (0,255,0), 2)
        cv2.rectangle(frame, (180+2*long,100+2*long), (180+3*long,100+3*long), (0,255,0), 2)
        #cv2.putText(frame,'Identify Results',(440,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(55,255,155),2)
        cv2.rectangle(frame, (46,75), (58,95), (0,0,0), -1)
        cv2.putText(frame,'A',(46,91),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (46,195), (58,215), (0,0,0), -1)
        cv2.putText(frame,'B',(46,211),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (46,315), (58,335), (0,0,0), -1)
        cv2.putText(frame,'C',(46,331),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (440,75), (452,95), (0,0,0), -1)
        cv2.putText(frame,'D',(440,91),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (440,195), (452,215), (0,0,0), -1)
        cv2.putText(frame,'E',(440,211),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (440,315), (452,335), (0,0,0), -1)
        cv2.putText(frame,'F',(440,331),cv2.FONT_HERSHEY_SIMPLEX,0.6,(55,255,155),2)
        cv2.rectangle(frame, (155,50), (500,72), (0,0,0), -1)
        cv2.rectangle(frame, (155,340), (430,360), (0,0,0), -1)
        cv2.rectangle(frame, (155,370), (430,390), (0,0,0), -1)
        frame=cv2ImgAddText(frame,'注:可点击识别色块改变颜色',160, 370, (55,255,155), 20)
        showTem2(tempList,frame)
        frame=tips(c,frame)
        cv2.namedWindow("real_time")
        cv2.setMouseCallback("real_time", mouse_click)
        frame=tips2(state,frame)
        cv2.imshow("real_time",frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            fflag=False
            break
        elif k == ord('s'):
            #存储
            #cv2.imwrite(r'./testgraph/'+str(c)+r'.jpg',frame)
            state=1
            img=frame
            hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            tempSolve=solve(hsv)
            tempList[c]=tempSolve
            print(tempSolve)
            num+=1
            #print('capture'+str(num))
            print('解析是否正确，若正确输入y,错误则进行重新按获取图像即可')
        elif k == ord('y'):
            if result[tempSolve[4]]=='#':
                state=2
                c+=1
                result[tempSolve[4]]=tempSolve
                print(tempList)
                cv2.imshow("real_time",frame)
                #print('当前存储的数据信息:')
                #print(result)
                #print('\n')
            else:
                state=3
                print('该面已经拍照,无需重复拍照')
        if check_dic(result):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(result)
    if fflag==True:
        return result
    else:
        return 'q'
