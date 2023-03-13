# 将ChFinAnn数据集的数据转化为 W2NER所需的格式
# 生成用于测试的数据集格式
# 全O标签只去五分之一，与BILSTM-CRF中dataSetTransCutLong.py数据处理中的保持一致
import json
import os

from tqdm import tqdm

roleAbbrDict = {'EquityHolder': 'EH', 'FrozeShares': 'FS', 'LegalInstitution': 'LI', 'StartDate': 'SD', 'EndDate': 'ED',
                'UnfrozeDate': 'UD', 'CompanyName': 'CN', 'HighestTradingPrice': 'HTP',
                'LowestTradingPrice': 'LTP', 'ClosingDate': 'CD', 'RepurchasedShares': 'RS', 'RepurchaseAmount': 'RA',
                'TradedShares': 'TS', 'AveragePrice': 'AP', 'LaterHoldingShares': 'LHS', 'Pledger': 'PR',
                'PledgedShares': 'PS', 'Pledgee': 'PE', 'ReleasedDate': 'RD', 'TotalPledgedShares': 'TPS',
                'TotalHoldingShares': 'THS', 'TotalHoldingRatio': 'THR'}  # 角色标签缩写字典
blackList = ['StockCode', 'StockAbbr', 'OtherType']  # 出现的其他无关类型
allCount = 0
notNoneCount = 0
NoneCount=0
saveNoneCount=0
saveCount = 0
for fileName in os.listdir("data/"):
    if '.json' in fileName:
        result = []  # 保存当前处理的结果，文件分析完毕后，写入此次结果
        with open('data/' + fileName, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            for data in tqdm(json_data):
                sentences = data[1]['sentences']
                for index, sent in enumerate(sentences):  # 以句子为处理单元
                    tempList = {}
                    nerList = []
                    ann_mspan2guess_field = data[1]['ann_mspan2guess_field']
                    #构建一个截断为100，超过100的进行句子切割，最后保留所有句子的数据集
                    if len(sent)>100:  #截断句子减少显存占用，提高batchsize
                        #1.截断
                        sent_sps_list=[]
                        sent_sps = sent.split('，')
                        for sent_sp in sent_sps:
                            if len(sent_sp) < 100:
                                sent_sps_list.append(sent_sp)
                            else:
                                sent_sps_list.append(sent_sp[:100])
                        allCount+=len(sent_sps)
                        #2.处理
                        for sent in sent_sps_list:
                            NoneFlag=True
                            tempList = {}
                            nerList = []
                            tempList["sentence"] = [i for i in sent]
                            for argument in ann_mspan2guess_field:
                                if argument in sent and ann_mspan2guess_field[argument] not in blackList:
                                    abbr = roleAbbrDict[ann_mspan2guess_field[argument]]
                                    # Equity Pledge事件中，Pledger在ann_mspan2guess_field表现为EquityHolder，需要判断一下,将其转化为PR
                                    if data[1]['recguid_eventname_eventdict_list'][0][1] == "EquityPledge" and abbr == 'EH':
                                        abbr = "PR"
                                    index_list=[str(sent).index(argument),str(sent).index(argument)+len(argument)]
                                    nerList.append({"index": [j for j in range(index_list[0], index_list[1])], "type": abbr})
                                    NoneFlag=False
                            if NoneFlag:
                                NoneCount += 1
                                if NoneCount%5==1:
                                    saveNoneCount+=1
                                    tempList["ner"] = nerList
                                    result.append(tempList)  # ensure_ascii=False, 中文字符不用unicode编码显示
                            else:
                                notNoneCount += 1
                                tempList["ner"] = nerList
                                result.append(tempList)  # ensure_ascii=False, 中文字符不用unicode编码显示
                    else:
                        allCount+=1
                        NoneFlag = True
                        tempList["sentence"] = [i for i in sent]
                        for argument in ann_mspan2guess_field:
                            if argument in sent and ann_mspan2guess_field[argument] not in blackList:
                                abbr = roleAbbrDict[ann_mspan2guess_field[argument]]
                                # Equity Pledge事件中，Pledger在ann_mspan2guess_field表现为EquityHolder，需要判断一下,将其转化为PR
                                if data[1]['recguid_eventname_eventdict_list'][0][1] == "EquityPledge" and abbr == 'EH':
                                    abbr = "PR"
                                for i in data[1]['ann_mspan2dranges'][argument]:  # 该论元出现的所有位置
                                    if i[0] == index:  # 属于本句的位置
                                        nerList.append({"index": [j for j in range(i[1], i[2])], "type": abbr})
                                NoneFlag = False
                        if NoneFlag:
                            NoneCount += 1
                            if NoneCount % 5 == 1:
                                saveNoneCount += 1
                                tempList["ner"] = nerList
                                result.append(tempList)  # ensure_ascii=False, 中文字符不用unicode编码显示
                        else:
                            notNoneCount += 1
                            tempList["ner"] = nerList
                            result.append(tempList)  # ensure_ascii=False, 中文字符不用unicode编码显示
        with open('../data/example/' + fileName, 'w+', encoding='utf-8') as file:
            saveCount+= len(result)
            file.write(json.dumps(result, ensure_ascii=False))
print("应保存实例数量", allCount,"非O标签",notNoneCount, "---全O标签数量",NoneCount, '---全O标签只保留为五分之一即', saveNoneCount, "--最终保存实例数量",
     saveCount)