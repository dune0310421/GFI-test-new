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


def update_dataset(token: str):
    gh = Github(token)
    with open("data/pandas.json", "r") as f:
        dataset = json.load(f)

    repo = gh.get_repo("pandas-dev/pandas")

    for item in dataset:
        logger.info("Processing #{}".format(item["number"]))
        if isinstance(item["resolved_in"], int): # resolved by a PR
            pr = request_github(gh, repo.get_pull, (item["resolved_in"],))
            item["changed_files"] = pr.changed_files
            item["additions"] = pr.additions
            item["deletions"] = pr.deletions
        else: # resolved by a commit
            commit = request_github(gh, repo.get_commit, (item["resolved_in"],))
            item["changed_files"] = len(commit.files)
            item["additions"] = commit.stats.additions
            item["deletions"] = commit.stats.deletions

    with open("data/pandas.json", "w") as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", help="GitHub token")
    args = parser.parse_args()
    update_dataset(token=args.token)
