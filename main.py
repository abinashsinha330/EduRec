import os
import time
import argparse
from tqdm import tqdm

from model import Model
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
student_chapters_dict = utils.get_student_chapters_sequences(args.dataset)

# calculate statistics related to dataset
num_students, num_chapters = utils.calculate_data_stats(student_chapters_dict)

# split data into training, validation and test sets
dataset = utils.train_valid_test_split(student_chapters_dict)
[student_train, student_valid, user_test] = dataset


num_batches = len(student_train) / args.batch_size
cc = 0.0
for u in student_train:
    cc += len(student_train[u])
print('Average sequence length in training set: %.2f' % (cc / len(student_train)))


sampler = utils.Sampler(student_train, num_students, num_chapters,
                        batch_size=args.batch_size, maxlen=args.max_len, n_workers=3)
model = Model(num_students, num_chapters, args)
# sess.run(tf.initialize_all_variables())

# code to run training given the number of epochs
T = 0.0
t0 = time.time()

try:
    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batches), total=num_batches, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print('Evaluating')
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print('')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
except:
    sampler.close()
    f.close()
    exit(1)

f.close()
sampler.close()
print("Done")
