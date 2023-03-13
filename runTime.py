def trans(N):
    min = 60
    hour = 60 * 60
    day = 60 * 60 * 24
    HOUR = N // hour
    MINUT = (N - (HOUR * hour)) // min
    SECONDS = N - ((HOUR * hour) + (MINUT * min))

    return '{}:{}:{}'.format(HOUR, MINUT, SECONDS)


#截断100 取十分之一
# train = 16050
# dev = 1991
# test = 1960

#截断100（舍弃）
# train = 160503
# dev = 19910
# test = 19594
#截断100 处理句子 保留五分一全O标签
train = 328358
dev = 47962
test = 45837
config = {"3090": {'price': 1.66, 'batchsize': 12,'trainDataProcess':8.82,'evalDataProcess':21},
          # "A100-PCIE-40GB": {'price': 3.54, 'batchsize': 8},
          # "A100-SMX4-80GB":{'price': 8.4, 'batchsize': 16,'trainDataProcess':5.6,'evalDataProcess':10}
          }

for i in config:
    price = config[i]['price']
    dataLoad = 300
    epoche = 10
    batchsize = config[i]['batchsize']
    trainDataProcess = config[i]['trainDataProcess']
    evalDataProcess=config[i]['evalDataProcess']
    dataLoadTime = (train + dev + test) / dataLoad
    trainTime = train / batchsize / trainDataProcess * epoche
    validTime = (dev + test) / batchsize / evalDataProcess * epoche
    testTime = test / batchsize / evalDataProcess
    allTime = dataLoadTime + trainTime + validTime + testTime
    print(i, "\n数据加载时间:", dataLoadTime, "总训练时间", trainTime, "总验证时间", validTime, "测试时间", testTime, "总时间", allTime,
          "格式化时间", trans(int(allTime)), "费用", (allTime / 3600 * price))
