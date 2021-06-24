import os
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Video')
parser.add_argument('--train_dir', default='train')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_len', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)


args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v)
                       for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

# prepare sequences of items for each user as per given user-item interactions
student_chapters_dict = utils.get_student_chapters_interactions(args.dataset)

# calculate statistics related to dataset
num_students, num_chapters = utils.calculate_data_stats(student_chapters_dict)

# split data into training, validation and test sets
dataset = utils.train_valid_test_split(student_chapters_dict)
[student_train, student_valid, student_test] = dataset


num_batches = len(student_train) / args.batch_size
cc = 0.0
for u in student_train:
    cc += len(student_train[u])
print('Average sequence length in training set: %.2f' % (cc / len(student_train)))
