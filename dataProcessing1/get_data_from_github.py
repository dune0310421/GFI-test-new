import pandas as pd
import csv
import pymongo
from pymongo import InsertOne,DeleteOne,UpdateOne
import matplotlib.pyplot as plt

import time
import json
import logging
import argparse

# !pip install PyGithub
from typing import *
from github import Github
from github import RateLimitExceededException, UnknownObjectException

T = TypeVar("T")
logger = logging.getLogger(__name__)

def connect_mongo(query={}, host='localhost', port=27017, username=None, password=None, db='test'):
    if username and password:
        mongo_uri = "mongodb://%s:%s@%s:%s/%s" % (username, password, host, port, db)
        client = pymongo.MongoClient(mongo_uri)
    else:
        client = pymongo.MongoClient(host, port)
    return client

def request_github(
        gh: Github, gh_func: Callable[..., T], params: Tuple = (), default: Any = None
) -> Optional[T]:
    """
    This is a wrapper to ensure that any rate-consuming interactions with GitHub
      have proper exception handling.
    """
    for _ in range(0, 3):  # Max retry 3 times
        try:
            data = gh_func(*params)
            return data
        except RateLimitExceededException as ex:
            logger.info("{}: {}".format(type(ex), ex))
            sleep_time = gh.rate_limiting_resettime - time.time() + 10
            logger.info("Rate limit reached, wait for {} seconds...".format(sleep_time))
            time.sleep(max(1.0, sleep_time))
        except UnknownObjectException as ex:
            logger.error("{}: {}".format(type(ex), ex))
            break
        except Exception as ex:
            logger.error("{}: {}".format(type(ex), ex))
            time.sleep(5)
    return default


def update_data(gh, item: dict) -> dict:
    dic = {}

    owner = item["owner"]
    name = item["name"]
    # repo = gh.get_repo(owner + "/" + name)
    repo = request_github(gh, gh.get_repo, (owner + "/" + name,))
    #     logger.info("Processing #{}".format(item["number"]))  # 打印日志信息

    if isinstance(item["resolved_in"], int):  # resolved by a PR
        pr = request_github(gh, repo.get_pull, (item["resolved_in"],))
        #         dic["changed_files"] = pr.changed_files
        #         dic["additions"] = pr.additions
        #         dic["deletions"] = pr.deletions
        #         dic["commit_num"] = pr.commits
        files = request_github(gh, pr.get_files)  # files = pr.get_files()
        dic["changed_files_list"] = list(map(lambda x: (x.filename, x.additions, x.deletions), files))

    else:  # resolved by a commit
        commit = request_github(gh, repo.get_commit, (item["resolved_in"],))
        #         dic["changed_files"] = len(commit.files)
        #         dic["additions"] = commit.stats.additions
        #         dic["deletions"] = commit.stats.deletions
        dic["changed_files_list"] = list(map(lambda x: (x.filename, x.additions, x.deletions), commit.files))

    # language = repo.language
    # dic["language"] = language

    return dic


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )  # 记录访问日志

    client = connect_mongo()
    collection = client.issues.first_issues
    token = "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"
    gh = Github(token)
    update_lst = []
    for data in collection.find().limit(5000):
        if 'language' in data.keys():
            dic = update_data(gh, data)
            update_lst.append(UpdateOne({"_id": data["_id"]}, {"$set": dic}))
    tmp1 = 1
    # res = collection.bulk_write(update_lst)