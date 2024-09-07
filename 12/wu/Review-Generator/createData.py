# creates data 
import string


reviewData = open("data/data.txt", 'r')
cleanData = open("data/cleanData.txt", 'w')
cleanDataPunc = open("data/cleanData_withPunc.txt", 'w')
cleanData.truncate(0)

onReview = 0
# 迭代每一行，将文本拆分为评论，并清除每一行的标点符号和大写字母
for line in reviewData:
    curLine = ""
    for word in line.split():
        if(word == "\"reviewText\":"):
            onReview = 1
        if(word == "\"summary\":" or word == "\"overall\":"):
            onReview = 0
            break
        if(onReview == 1):
            curLine += word + " "
            
    # 删除新字符行
    curLine = curLine.replace('\\n', ' ')
    curLine = curLine.replace('  ', ' ')


    cleanDataPunc.write(curLine[15:-3])
    cleanDataPunc.write("\n")
    
    #去掉标点符号
    curLine = curLine.translate(str.maketrans('', '', string.punctuation))

    #修剪线的起点和终点
    cleanData.write(curLine[11:])

    #添加换行符
    cleanData.write("\n")

cleanData.close()