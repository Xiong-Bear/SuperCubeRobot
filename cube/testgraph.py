from cube.parse import graph
from cube.parse import dataParse
import cv2

if __name__=='__main__':
    tempSolve=['X','X','X','X','X','X','X','X','X']
    result = {'O':'#','B':'#','R':'#','Y':'#','W':'#','G':'#'}
    c=0
    for i in range(0,6):
        img=cv2.imread(r'./testgraph/'+str(i)+'.jpg')
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        tempSolve=graph.solve(hsv)
        result[tempSolve[4]]=tempSolve
        print(tempSolve)
    while True:
        try:
            step_str = dataParse.parse(result)
            print(step_str)
        except UserWarning:
            flag=1
            print('拍照有误，请重新拍照')

        else :
            # fo = open("code/step.h", "w")
            # str = "char step[] = \"" + step_str + "\";"
            # fo.write(str)
            exit(0)
