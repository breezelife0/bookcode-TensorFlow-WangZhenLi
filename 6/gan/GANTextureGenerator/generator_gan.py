
import math
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from image import ImageVariations
from network import image_decoder, image_encoder, image_output, gan_optimizer

LOG_DIR = 'logs'

class GANetwork():

    def __init__(self, name, setup=True, image_size=64, colors=3, batch_size=64, directory='network', image_manager=None,
                 input_size=64, learning_rate=0.0002, dropout=0.4, generator_convolutions=5, generator_base_width=32,
                 discriminator_convolutions=4, discriminator_base_width=32, classification_depth=1, grid_size=4,
                 log=True, y_offset=0.1, learning_momentum=0.6, learning_momentum2=0.9, learning_pivot=10000,
                 dicriminator_scaling_favor=3):
        """
        创建用于生成图像的GAN
        """
        self.name = name
        self.image_size = image_size
        self.colors = colors
        self.batch_size = batch_size
        self.grid_size = min(grid_size, int(math.sqrt(batch_size)))
        self.log = log
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        #网络变量
        self.input_size = input_size
        self._gen_conv = generator_convolutions
        self._gen_width = generator_base_width
        self._dis_conv = discriminator_convolutions
        self._dis_width = discriminator_base_width
        self._class_depth = classification_depth
        self._dropout = dropout
        #训练变量
        self.learning_rate = (learning_rate, learning_momentum, learning_momentum2, learning_pivot)
        self._y_offset = y_offset
        self.current_scale = 1.0
        self._dis_scale = dicriminator_scaling_favor
        #设置图像
        if image_manager is None:
            self.image_manager = ImageVariations(image_size=image_size, colored=(colors == 3))
        else:
            self.image_manager = image_manager
            self.image_manager.image_size = image_size
            self.image_manager.colored = (colors == 3)
        #设置网络
        self.iterations = tf.Variable(0, name="training_iterations", trainable=False)
        with tf.variable_scope('input'):
            self.generator_input = tf.placeholder(tf.float32, [None, self.input_size], name='generator_input')
            self.image_input = tf.placeholder(tf.uint8, shape=[None, image_size, image_size, self.colors], name='image_input')
            self.image_input_scaled = tf.subtract(tf.to_float(self.image_input)/127.5, 1, name='image_scaling')
        self.generator_output = None
        self.image_output = self.image_grid_output = None
        self.generator_solver = self.discriminator_solver = self.scale = None
        if setup:
            self.setup_network()

    def setup_network(self):
        """如果未在构造函数中初始化网络，则初始化网络"""
        self.generator_output = image_decoder([self.generator_input], 'generator', self.image_size, self._gen_conv, self._gen_width, self.input_size, self.batch_size, self.colors)[0]
        self.image_output, self.image_grid_output = image_output([self.generator_output], 'output', self.image_size, self.grid_size)
        gen_logit, image_logit = image_encoder([self.generator_output, self.image_input_scaled], 'discriminator', self.image_size, self._dis_conv, self._dis_width, self._class_depth, self._dropout, 1)
        gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.generator_solver, self.discriminator_solver, self.scale = \
            gan_optimizer('train', gen_var, dis_var, gen_logit, image_logit, 0., 1-self._y_offset,
                          *self.learning_rate, self.iterations, self._dis_scale, summary=self.log)


    def random_input(self):
        """为生成器创建随机输入"""
        return np.random.uniform(0.0, 1.0, size=[self.batch_size, self.input_size])


    def generate(self, session, name, amount=1):
        """生成一个图像并保存它"""
        def get_arr():
            arr = np.asarray(session.run(
                self.image_output,
                feed_dict={self.generator_input: self.random_input()}
            ), np.uint8)
            arr.shape = self.batch_size, self.image_size, self.image_size, self.colors
            return arr
        if amount == 1:
            self.image_manager.save_image(get_arr()[0], name)
        else:
            images = []
            counter = amount
            while counter > 0:
                images.extend(get_arr())
                counter -= self.batch_size
            for i in range(amount):
                self.image_manager.save_image(images[i], "%s_%02d"%(name, i))

    def generate_grid(self, session, name):
        """生成一个图像并保存它"""
        grid = session.run(
            self.image_grid_output,
            feed_dict={self.generator_input: self.random_input()}
        )
        self.image_manager.image_size = self.image_grid_output.get_shape()[1]
        self.image_manager.save_image(grid, name)
        self.image_manager.image_size = self.image_size


    def get_session(self):
        saver = tf.train.Saver()
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        try:
            saver.restore(session, os.path.join(self.directory, self.name))
            start_iteration = session.run(self.iterations)
            print("\n已加载现有网络\n")
        except Exception as e:
            start_iteration = 0
            if self.log:
                print("\n创建了一个新网络 (%s)\n"%repr(e))
        return session, saver, start_iteration


    def __get_feed_dict__(self):
        return {
            self.image_input: self.image_manager.get_batch(self.batch_size),
            self.generator_input: self.random_input()
        }

    def __training_iteration__(self, session, i):
        if i < 500:                     #迭代初始化
            if i < 50:
                session.run([self.discriminator_solver], feed_dict=self.__get_feed_dict__())
            else:
                session.run([self.discriminator_solver, self.generator_solver], feed_dict=self.__get_feed_dict__())
        elif i%10 == 0:                 #检查缩放
            _, _, self.current_scale = session.run([self.discriminator_solver, self.generator_solver, self.scale], feed_dict=self.__get_feed_dict__())
        elif self.current_scale > 1.3:  #只训练性能较差的网络（进行一些额外的更快的迭代）
            session.run(self.generator_solver, feed_dict={self.generator_input: self.random_input()})
            session.run(self.generator_solver, feed_dict={self.generator_input: self.random_input()})
        elif self.current_scale > 0.7:  #如果利润在30%以内，则对两个网络进行培训
            session.run([self.discriminator_solver, self.generator_solver], feed_dict=self.__get_feed_dict__())
        else:                           #只训练表现较差的网络
            session.run([self.discriminator_solver], feed_dict=self.__get_feed_dict__())


    def train(self, batches=100000, print_interval=10):
        """为多个批次培训网络（如果有现有型号，则继续）"""
        start_time = last_time = last_save = timer()
        session, saver, start_iteration = self.get_session()
        if self.log:
            logger = SummaryLogger(self, session, start_iteration)
        try:
            print("对GAN进行图像训练'%s' folder"%self.image_manager.in_directory)
            print("要提前停止训练，请按Ctrl+C（将保存进度）")
            print('要继续训练，只需再次运行训练')
            if self.log:
                print("要查看进度，请运行'python -m tensorflow.tensorboard --logdir %s'"%LOG_DIR)
            print("要使用经过训练的网络生成图像，请运行 'python generate.py %s'"%self.name)
            print()
            time_per = 10
            for i in range(start_iteration+1, start_iteration+batches+1):
                self.__training_iteration__(session, i)
                session.run(self.iterations.assign(i))
                #打印进度
                if i%print_interval == 0:
                    curr_time = timer()
                    time_per = time_per*0.6 + (curr_time-last_time)/print_interval*0.4
                    time = curr_time - start_time
                    print("\rIteration: %04d    Time: %02d:%02d:%02d  (%02.1fs / iteration)" % \
                        (i, time//3600, time%3600//60, time%60, time_per), end='')
                    last_time = curr_time
                if self.log:
                    logger(i)
                #保存网络
                if timer() - last_save > 1800:
                    saver.save(session, os.path.join(self.directory, self.name), self.iterations)
                    last_save = timer()
        except KeyboardInterrupt:
            pass
        finally:
            print()
            if self.log:
                logger.close()
            print("保存网络")
            saver.save(session, os.path.join(self.directory, self.name))
            session.close()


class SummaryLogger():
    """将训练进度记录到tensorboard（以及一些进度输出到控制台）"""
    def __init__(self, network, session, iteration, summary_interval=20, image_interval=500):
        self.session = session
        self.gan = network
        self.image_interval = image_interval
        self.summary_interval = summary_interval
        os.makedirs(LOG_DIR, exist_ok=True)
        if iteration == 0:
            self.writer = tf.summary.FileWriter(os.path.join(LOG_DIR, network.name), session.graph)
        else:
            self.writer = tf.summary.FileWriter(os.path.join(LOG_DIR, network.name))
        self.summary = tf.summary.merge_all()
        self.batch_input = network.random_input()

    def __call__(self, iteration):
        #保存图像
        if iteration%self.image_interval == 0:
            #使tensorboard显示多个图像，而不仅仅是最新的图像
            feed_dict = self.gan.__get_feed_dict__()
            feed_dict[self.gan.generator_input] = self.batch_input
            image, summary = self.session.run(
                [tf.summary.image(
                    'training/iteration/%d'%iteration,
                    tf.stack([self.gan.image_grid_output]),
                    max_outputs=1,
                    collections=['generated_images']
                ), self.summary],
                feed_dict=feed_dict
            )
            self.writer.add_summary(image, iteration)
            self.writer.add_summary(summary, iteration)
        elif iteration%self.summary_interval == 0:
            feed_dict = self.gan.__get_feed_dict__()
            #保存摘要
            summary = self.session.run(self.summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, iteration)

    def close(self):
        self.writer.close()

