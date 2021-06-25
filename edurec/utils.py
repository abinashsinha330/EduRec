"""
Script containing utility functions:
1. calculate_data_stats:
    calculate stats pertaining to sequential data like:
    1. average length of sequences
    2. maximum length of sequences
    3. minimum length of sequences
    4. number of sequences
    5. number of unique items across all sequences
2. train_valid_test_split:
    split data into training, validation and test sets
3. get_student_chapters_sequences
    form the sequences of chapters for each of all students
"""

import sys
import numpy as np
from multiprocessing import Queue, Process
from collections import defaultdict


def calculate_data_stats(user_items_dict):
    """
    Method to calculate statistics related to data

    :param user_items_dict:
    :return:
    """
    items_set = set()
    sum_num_items = 0
    max_num_items = -1
    min_num_items = sys.maxsize
    for user in user_items_dict:
        curr_items_list = user_items_dict[user]
        items_set.update(curr_items_list)
        curr_num_items = len(curr_items_list)
        sum_num_items += curr_num_items
        if curr_num_items > max_num_items:
            max_num_items = curr_num_items
        if curr_num_items < min_num_items:
            min_num_items = curr_num_items

    num_users = len(user_items_dict)
    num_items = len(items_set)
    print('There are {} users!'.format(num_users))
    print('There are {} unique items across all users!'.format(num_items))
    print('{} is average number of items each user interacts with!'.format(
        round(sum_num_items / num_users, 2))
    )
    print('{} is maximum number of items any user interacts with!'.format(
        max_num_items)
    )
    print('{} is minimum number of items any user interacts with!'.format(
        min_num_items)
    )

    return num_users, num_items


def train_valid_test_split(user_items_dict):
    """
    Method to split data into training, validation and test
    sets; select second last item for every user to be in validation
    set and last item for every user in test set, if less than 3 items
    for any user just leave validation & test sets empty for that user

    :param fname:
    :return:
    """
    user_train = {}
    user_valid = {}
    user_test = {}
    for user in user_items_dict:
        nfeedback = len(user_items_dict[user])
        if nfeedback < 3:
            user_train[user] = user_items_dict[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = user_items_dict[user][:-2]
            user_valid[user] = []
            user_valid[user].append(user_items_dict[user][-2])
            user_test[user] = []
            user_test[user].append(user_items_dict[user][-1])

    return [user_train, user_valid, user_test]


def get_student_chapters_sequences(fname):
    # usernum = 0
    # itemnum = 0
    user_items_dict = defaultdict(list)
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.strip().split(' ')
        u = int(u)
        i = int(i)
        # usernum = max(u, usernum)
        # itemnum = max(i, itemnum)
        user_items_dict[u].append(i)
    return user_items_dict


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_fn(student_chapters_dict, num_students, num_chapters,
              batch_size, max_len, result_queue, seed):
    def sample():

        student = np.random.randint(1, num_students + 1)
        while len(student_chapters_dict[student]) <= 1:
            student = np.random.randint(1, num_students + 1)

        seq = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len], dtype=np.int32)
        nxt = student_chapters_dict[student][-1]
        idx = max_len - 1

        uniq_chapters = set(student_chapters_dict[student])
        for i in reversed(student_chapters_dict[student][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, num_chapters + 1, uniq_chapters)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return student, seq, pos, neg

    np.random.seed(seed)
    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class Sampler(object):
    def __init__(self, user_items_dict, num_users, num_items,
                 batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_fn, args=(user_items_dict,
                                                num_users,
                                                num_items,
                                                batch_size,
                                                maxlen,
                                                self.result_queue,
                                                np.random.randint(2e9)
                                                )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
