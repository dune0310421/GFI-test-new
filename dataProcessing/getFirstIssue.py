import dateutil
import copy

def get_first_issue(data):
    cls_id_dict = {}
    # 按cls_id(issue resolver)建立字典，以便提取first issue
    for item in data:
        #     print(item['issue_id'])
        if item['cls_id'] not in cls_id_dict.keys():
            cls_id_dict[item['cls_id']] = [item['clst']]
        else:
            cls_id_dict[item['cls_id']].append(item['clst'])
    for key in cls_id_dict.keys():
        cls_id_dict[key].sort()
    print("issue cnt in total: " + str(len(data)))
    print("num of resolver: " + str(len(cls_id_dict.keys())))

    # 分离LTC和OTC，分别将id存入列表中
    LTC_id = []
    OTC_id = []
    # cnt = 0
    for key in cls_id_dict.keys():
        if len(cls_id_dict[key]) >= 2:
            #         print(key)
            #         first = cls_id_dict[key][0]
            #         last = cls_id_dict[key][-1]
            #         d1 = dateutil.parser.parse(first)
            #         d2 = dateutil.parser.parse(last)
            #         if ((d2-d1).days > 365):
            LTC_id.append(key)
        #         else:
        #             OTC_id.append(key)
        else:
            OTC_id.append(key)
            # cnt += 1
    # print(LTC_id)
    # print(cnt)

    # 分离LTC和OTC issues
    LTC_issues = []
    OTC_issues = []
    for item in data:
        if item['cls_id'] in LTC_id:
            LTC_issues.append(item)
        elif item['cls_id'] in OTC_id:
            OTC_issues.append(item)
        else:
            print(item['cls_id'])
    print("issue cnt by LTC: " + str(len(LTC_issues)))
    print("issue cnt by OTC: " + str(len(OTC_issues)))

    # 提取第一个issue
    LTC_1st_issues = []
    OTC_1st_issues = []
    tmp_id_1 = copy.copy(LTC_id)
    tmp_id_2 = copy.copy(OTC_id)
    for item in data:
        #     print(item['cls_id'])
        if item['cls_id'] in tmp_id_1:
            if item['clst'] == cls_id_dict[item['cls_id']][0]:
                LTC_1st_issues.append(item)
                tmp_id_1.remove(item['cls_id'])
        elif item['cls_id'] in tmp_id_2:
            if item['clst'] == cls_id_dict[item['cls_id']][0]:
                OTC_1st_issues.append(item)
                tmp_id_2.remove(item['cls_id'])
    #     else:
    #         print(item['cls_id'])
    print("first issue cnt by LTC: " + str(len(LTC_1st_issues)))
    print("first issue cnt by OTC: " + str(len(OTC_1st_issues)))
    # print(LTC_1st_issues[0])

    return LTC_1st_issues, OTC_1st_issues




