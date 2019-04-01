import tensorflow as tf
#
# g1=tf.Graph()
# #v=tf.Variable(name='v',initial_value=tf.zeros(shape=[1]))
# with g1.as_default():
#     v=tf.get_variable("v",initializer=tf.zeros(shape=[2,3]))
# with tf.Session(graph=g1) as sess:
#     #tf.initialize_all_variables().run()
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable("v")))
#
#
# a=tf.constant([1.0,2.0],name="a")
# b=tf.constant([3.0,4.0],name="b")
# tf.Variable(name='v1',initial_value=tf.ones(shape=[3,4]))
# result=tf.multiply(a,b,name="c")
# print(result.get_shape())
# with tf.Session() as sess:
#     m=sess.run(result)
#     n=result.eval()#tf.Tensor.eval()
#     print(n)



# ###########front#######
# n=2
# x=tf.placeholder(tf.float32,shape=[n,2],name='input')
# w1=tf.Variable(tf.random_normal([2,3],stddev=2))
# w2=tf.Variable(tf.random_normal([3,1],stddev=1))
# a1=tf.matmul(x,w1)
# result=tf.matmul(a1,w2)
#
# with tf.Session() as sess:
#     init_op=tf.initialize_all_variables()
#     sess.run(init_op)
#     print(sess.run(result,feed_dict={x:[[1,2],[3,4]]}))
#
# ########################


# #####Loss##################################################
# cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(result,1e-10,1.0)))
# ####learning rate####
# learning_rate=0.001
# train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# ############################################################

# ####for example#############
# import  numpy as np
#
# X_=np.random.random_sample((1500,2))
# #print(x)
# #y_=np.zeros(shape=[5000])
# #for i in range(len(x_)):
#     #y_[i]=int(((x_[i][0]+x_[i][1])>1))
# Y_=[[int(x1+x2>1)] for (x1,x2) in X_]
# #print(y)
# #print(Y)
# print(X_)
# print(Y_)
# batch_size=8
#
# x=tf.placeholder(shape=(None,2),dtype=tf.float32, name="input")
# y=tf.placeholder(shape=(None,1),dtype=tf.float32, name='output')
# w1=tf.Variable(tf.random_normal(shape=[2,3],stddev=2,seed=2))
# w2=tf.Variable(tf.random_normal(shape=[3,1],stddev=1,seed=1))
# a1=tf.matmul(x,w1)
# result=tf.matmul(a1,w2)
#
# cross_entropy=-tf.reduce_mean(y*tf.log(tf.clip_by_value(result,1e-10,1.0)))
# learning_rate=0.001
# train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#
# with tf.Session() as sess:
#     init_op=tf.initialize_all_variables()
#     sess.run(init_op)
#
#     print("before training:")
#     print(sess.run(w1))
#     print(sess.run(w2))
#
#     step=500
#     for i in range(step):
#         start=(i*batch_size)%1500
#         end=min(start+batch_size,1500)
#
#         sess.run(train_step,feed_dict={x:X_[start:end],y:Y_[start:end]})
#
#         if i%20==0:
#             all_cross_entropy=sess.run(cross_entropy,feed_dict={x:X_,y:Y_})
#             print("After %d training steps, cross entropy on all data is %g" %(i,all_cross_entropy))
#
#
#     print("after training")
#     print(sess.run(w1))
#     print(sess.run(w2))
# ########################################################


#   ###############################################
# a=tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
# with tf.Session() as sess:
#     print(sess.run(tf.reduce_mean(a)))
#   #tf.reduce_mean()求整个矩阵的均值（即所有值均值）
#   ##################################################



# #############################3
# global_step=tf.Variable(0)
# learning_rate=tf.train.exponential_decay(0.1,global_step,300,0.96) ####(learning_rate:初始值, global_step, decay_step:衰减速度，多少次循环后衰减一次，decay_rate:衰减率, staircase=TRUE阶梯化;)
# learning_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step)
# #####初始学习率0.1，每迭代300次，学习率衰减0.96，阶梯化（global_step/decay_step取整）


#
# def get_weight(shape,lamda):
#     var=tf.Variable(tf.random_normal(shape,dtype=tf.float32,stddev=1))
#     tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamda)(var))
#     loss_c=tf.contrib.layers.l2_regularizer(lamda)(var)
#     return var,loss_c
#
#
# y_=[7]
# x=tf.placeholder(shape=[None,2],name='input',dtype=tf.float32)
# y=tf.placeholder(dtype=tf.float32,shape=[None,1],name='output')
#
# lamda=0.05
# dimension=[2,3,4,4,1]
# n_layers=len(dimension)
# current_layer=x
# in_dimension=dimension[0]
# loss_c=[0.0]*4
#
# for i in range(1,n_layers):
#     out_dimension=dimension[i]
#     weights,loss_c[i-1]=get_weight([in_dimension,out_dimension],lamda)
#     bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
#     current_layer=tf.nn.relu(tf.matmul(current_layer,weights)+bias)
#     in_dimension=out_dimension
#
# mse_loss=tf.reduce_mean(tf.square(y_-current_layer))
#
# tf.add_to_collection('losses',mse_loss)
# loss=tf.add_n(tf.get_collection('losses')) #get_collection 返回一个名为‘losses’的列表，里面所有元素组成的列表
#
# with tf.Session() as sess:
#     init_op=tf.initialize_all_variables()
#     sess.run(init_op)
#     print(sess.run(loss,feed_dict={x:[[1,3]],y:[[5]]}))
#     #print(sess.run(loss_c,feed_dict={x:[[1,3]],y:[[5]]}))
#     #print(sess.run(mse_loss,feed_dict={x:[[1,3]],y:[[5]]}))



























