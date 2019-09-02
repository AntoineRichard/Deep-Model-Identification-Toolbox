import numpy as np
import tensorflow as tf

config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)


#sess = tf.Session()
saver = tf.train.import_meta_graph("output_test/best_NN.meta")
saver.restore(sess, "output_test/best_NN")

W1 = sess.run(tf.get_default_graph().get_tensor_by_name("dense1/kernel:0"))
B1 = sess.run(tf.get_default_graph().get_tensor_by_name("dense1/bias:0"))

W2 = sess.run(tf.get_default_graph().get_tensor_by_name("dense2/kernel:0"))
B2 = sess.run(tf.get_default_graph().get_tensor_by_name("dense2/bias:0"))

W3 = sess.run(tf.get_default_graph().get_tensor_by_name("output/kernel:0"))
B3 = sess.run(tf.get_default_graph().get_tensor_by_name("output/bias:0"))

np.savetxt("output_test/W1.csv", W1, delimiter=",")
np.savetxt("output_test/B1.csv", B1, delimiter=",")
np.savetxt("output_test/W2.csv", W2, delimiter=",")
np.savetxt("output_test/B2.csv", B2, delimiter=",")
np.savetxt("output_test/W3.csv", W3, delimiter=",")
np.savetxt("output_test/B3.csv", B3, delimiter=",")

np.savetxt("output_test/means.csv", np.load("output_test/means.npy"), delimiter=",")
np.savetxt("output_test/std.csv", np.load("output_test/std.npy"), delimiter=",")
