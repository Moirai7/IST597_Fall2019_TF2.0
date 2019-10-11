# IST597_Fall2019_TF2.0
# Official github page for all IST597 Assignments 
* Refer to official webpage for latest updates[http://clgiles.ist.psu.edu/IST597/index.html]
* Starter code and Problem set for second assignment[Assignment #00010]
* Fork the repo to stay updated on latest commit 

tensorflow2.0
很多功能转移到keras的框架下，变得更加简洁。
去掉了global graph/variables的概念。取variables需要用python的object而不是使用name。因此在设计model的时候，可能存在一些麻烦。比如之前可以在需要变量的时候get_variable，variable是reuse并且trainable就可以。现在不能get_variable，每次tf.Variable就会添加一个新的变量而不是重用之前的变量，因为他对应了一个新的object。所以要先定义好variable object，再使用object。但是这个对于有很多相似层，想用variable_scope来重复使用代码不友好。但是如果使用keras，因为大部分layers都实现好了，所以可以不用申请variables，可能会方便。
第二，不能使用tf.trainable_variables。需要构建module或者keras.model，再用model取变量。
第三，tf.save_model需要使用signature定义function，并且module不能再使用trainable_variables。可能keras可以使用未测试。但就目前设计，想继续训练很麻烦。
第四，以前版本如conv2d，batchnorm在2.0上被删减，如果不使用keras，调用nn里的接口很复杂

第五，原来multirnn，先选择rnncell，然后multirnn，在dynamic run返回的维度是3D（可能中间cell输入还是2D）。现在rnncell输入是2D，stack后输入还是2D，如果3D的话就维度错误。但是不用rnncell，选择rnn就可以输入3D
