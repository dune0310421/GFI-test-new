import time
import json
import logging
import argparse

from typing import *
from github import Github
from github import RateLimitExceededException, UnknownObjectException

T = TypeVar("T")
logger = logging.getLogger(__name__)


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


def update_dataset(filepth: str, token: str):
    with open(filepth, "r") as f:
        dataset = json.load(f)
    update(dataset)

def update(dataset: list):
    gh = Github(token)
    for item in dataset:
        owner = item["owner"]
        name = item["name"]
        repo = gh.get_repo(owner+"/"+name)
        logger.info("Processing #{}".format(item["number"]))  # 打印日志信息
        if isinstance(item["resolved_in"], int):  # resolved by a PR
            pr = request_github(gh, repo.get_pull, (item["resolved_in"],))
            # commits = pr.get_commits().get_page(0)
            # item["changed_files_list"] = []
            # i = 0
            # files = pr.get_files().get_page(i)
            # while files != []:
            #     for item1 in files:
            #         tmp_dic = {}
            #         tmp_dic['file_name'] = item1.filename
            #         # tmp_dic['file_additions'] = item1.additions
            #         # tmp_dic['file_deletions'] = item1.deletions
            #         # tmp_dic['pre_file'] = item1.previous_filename
            #         item["changed_files_list"].append(tmp_dic)
            #     i = i + 1
            #     files = pr.get_files().get_page(i)
            item["changed_files"] = pr.changed_files
            item["additions"] = pr.additions
            item["deletions"] = pr.deletions
            item["commit_num"] = pr.commits
        else:  # resolved by a commit
            commit = request_github(gh, repo.get_commit, (item["resolved_in"],))
            item["changed_files"] = len(commit.files)
            # item["changed_files_list"] = list(map(lambda x: x.filename, commit.files))
            # print(item["changed_files_list"])
            item["additions"] = commit.stats.additions
            item["deletions"] = commit.stats.deletions

    # with open("../data/pandas-dataset.json", "w") as f:
    #     json.dump(dataset, f, indent=2)
    with open("../data/pandas_gfi_pr.json", "w") as f1:
        json.dump(dataset, f1, indent=2)
    # with open("../data/pandas_gfi_commit1.json", "w") as f2:
    #     json.dump(dataset, f2, indent=2)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )  # 记录访问日志

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--token", help="GitHub token")
    # args = parser.parse_args()
    # update_dataset(token=args.token)
    token = "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"
    update_dataset(filepth="../data/pandas_gfi_pr.json", token=token)
