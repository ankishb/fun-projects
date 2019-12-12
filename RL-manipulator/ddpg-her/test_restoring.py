import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.Session()
new_saver = tf.train.import_meta_graph('./model/kk.meta')
print(new_saver)
new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)