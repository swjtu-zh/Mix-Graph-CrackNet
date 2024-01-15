import tensorflow as tf
from einops import rearrange
import numpy as np

class GraphBasedSkipConnection (tf.keras.layers.Layer):

    def __init__(self, input_channels=512):
        super(GraphBasedSkipConnection, self).__init__()

        self.input_channels = input_channels

        self.edge_aggregation_func = tf.keras.Sequential([tf.keras.layers.Dense(1, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()], name='test1')

        self.vertex_update_func = tf.keras.Sequential([tf.keras.layers.Dense(input_channels // 2, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()], name='test2')

        self.edge_update_func = tf.keras.Sequential([tf.keras.layers.Dense(input_channels // 2, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()], name='test3')

        self.update_edge_reduce_func = tf.keras.Sequential([tf.keras.layers.Dense(1, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()], name='test4')

        self.final_aggregation_layer = tf.keras.Sequential([tf.keras.layers.Conv2D(input_channels, kernel_size=1,
                                                                use_bias=False, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU()], name='test5')

    def call(self, input):
        x = input
        shape = x.shape
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        # print(B)
        vertex = input
        vertex = rearrange(vertex, "b h w c -> b c h w")

        edge = tf.stack(
            [
                tf.keras.layers.Concatenate(axis=2)([vertex[:, :, -1:], vertex[:, :, :-1]]),
                tf.keras.layers.Concatenate(axis=2)([vertex[:, :, 1:], vertex[:, :, :1]]),
                tf.keras.layers.Concatenate(axis=3)([vertex[:, :, :, -1:], vertex[:, :, :, :-1]]),
                tf.keras.layers.Concatenate(axis=3)([vertex[:, :, :,  1:], vertex[:, :, :,  :1]])
            ], axis=-1
        ) * tf.expand_dims(vertex, axis=-1)

        # aggregated_edge = self.edge_aggregation_func(rearrange(edge, "b c h w n -> (b c h w) n"))
        aggregated_edge = self.edge_aggregation_func(rearrange(edge, "b c h w n -> b (c h w) n"))
        # print(aggregated_edge.shape)
        # aggregated_edge = tf.reshape(aggregated_edge, [B, C, H, W])
        aggregated_edge = rearrange(aggregated_edge, "b (c h w) n -> b c h w n", c=C, h=H, w=W)
        aggregated_edge = tf.squeeze(aggregated_edge, axis=-1)

        cat_feature_for_vertex = tf.keras.layers.Concatenate(axis=1)([vertex, aggregated_edge])

        update_vertex = self.vertex_update_func(rearrange(cat_feature_for_vertex, "b c h w -> b (h w) c", c=2 * self.input_channels))
        update_vertex = rearrange(update_vertex, "b (h w) c -> b c h w",  h=H, w=W, c=self.input_channels // 2)

        cat_feature_for_edge = tf.keras.layers.Concatenate(axis=1)([tf.stack([vertex, vertex, vertex, vertex], axis=-1), edge])
        cat_feature_for_edge = rearrange(cat_feature_for_edge, "b c h w n -> b (h w n) c", c=2 * self.input_channels)

        update_edge = self.edge_update_func(cat_feature_for_edge)
        update_edge = rearrange(update_edge, "b (h w n) c -> b (c h w) n",  h=H, w=W, n=4, c=C//2)

        update_edge_converted = self.update_edge_reduce_func(update_edge)
        update_edge_converted = rearrange(update_edge_converted, "b (c h w) n -> b c h w n", c=C//2, h=H, w=W)
        update_edge_converted = tf.squeeze(update_edge_converted, axis=-1)

        update_feature = update_vertex * update_edge_converted

        output = self.final_aggregation_layer(tf.keras.layers.Concatenate()([x, rearrange(update_feature, "b c h w -> b h w c")]))

        return output

class TransformerAttentionModule(tf.keras.layers.Layer):

    def __init__(self, channel):
        super(TransformerAttentionModule, self).__init__()
        self.channel = channel
        self.conv = tf.keras.layers.Conv2D(channel,kernel_size=3, padding="same")


    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Reshape((1, 1, self.channel))(x)
        attn = tf.keras.layers.Activation('sigmoid')(x)
        out_feature = tf.keras.layers.multiply([inputs, attn])
        return out_feature

class ConvolutionAttentionModule(tf.keras.layers.Layer):
    def __init__(self, channel):
        super(ConvolutionAttentionModule, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(channel, kernel_size=(3,1), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(channel, kernel_size=(1, 3), padding="same")
        self.proj = tf.keras.layers.Conv2D(1, kernel_size=1)
    def call(self, input):
        shape = input.shape
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        branch1 = tf.keras.layers.MaxPooling2D(pool_size=(1, W), strides=(1, W))(input) #(b, h, 1, c)
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(H, 1), strides=(H, 1))(input) #(b, 1, w, c)
        branch1 = self.conv1(branch1)
        branch2 = self.conv2(branch2)
        branch1 = tf.keras.layers.UpSampling2D(size=(1, W))(branch1)
        branch2 = tf.keras.layers.UpSampling2D(size=(H, 1))(branch2)
        feature = tf.keras.layers.add([branch1, branch2])
        feature = self.proj(feature)
        attn = tf.keras.layers.Activation('sigmoid')(feature)
        out_feature = tf.keras.layers.multiply([input, attn])
        return out_feature


class ConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(ConvBnRelu, self).__init__()
        self.fil_num = fil_num
        self.conv = tf.keras.layers.Conv2D(filters=fil_num, kernel_size=3, padding='same', activation=None)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class DeConvBnRelu(tf.keras.layers.Layer):
    def __init__(self, fil_num):
        super(DeConvBnRelu, self).__init__()
        self.fil_num = fil_num
        self.conv = tf.keras.layers.Conv2DTranspose(filters=fil_num, kernel_size=3, strides=(2, 2), padding='same', activation=None)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ConvolutionalBranch(tf.keras.layers.Layer):

    def __init__(self, dim):
        super().__init__()
        # self.conv = tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=False)
        self.proj = tf.keras.layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.act = tf.keras.layers.Activation("relu")
        self.proj1 = tf.keras.layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.act1 = tf.keras.layers.Activation("relu")
    def call(self, x):
        # x = self.conv(inputs)
        x = self.proj(x)
        x = self.act(x)
        x = self.proj1(x)
        x = self.act1(x)
        return x


class TransformerBranch(tf.keras.layers.Layer):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim = int(dim // num_heads)
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.pool_size = pool_size
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(pool_size, pool_size)) \
            if pool_size > 1 else tf.keras.layers.Activation('linear')
        self.uppool = tf.keras.layers.UpSampling2D((pool_size, pool_size), interpolation='bilinear')\
            if pool_size > 1 else tf.keras.layers.Activation('linear')

    def att_fun(self, q, k, v, B, N, C):

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = tf.matmul(attn, v)
        x = rearrange(x, "b h n c ->  b n h c")
        x = rearrange(x, " b n h c ->  b n (h c)"  )
        # x = tf.reshape(tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3]), [B, N, C])

        return x

    def call(self, x):
        # B, H, W, C
        shape = x.shape
        B, H, W, _ = shape[0], shape[1], shape[2], shape[3]
        xa = self.pool(x)
        xa = rearrange(xa, "b h w c -> b (h w) c")
        # xa = tf.reshape(xa, [B, -1, self.dim])
        shape = xa.shape
        B, N, C = shape[0], shape[1], shape[2]
        qkv = self.qkv(xa)
        qkv = rearrange(qkv, "b n (m h nh) -> b n m h nh",  m=3, h=self.num_heads,nh=int(C//self.num_heads))
        qkv = rearrange(qkv, "b n m h nh -> m b h n nh")
        # qkv = tf.transpose(tf.reshape(self.qkv(xa), [B, N, 3, self.num_heads, int(C//self.num_heads)]), [2, 0, 3, 1, 4])

        q, k, v = qkv[0], qkv[1], qkv[2]
        xa = self.att_fun(q, k, v, B, N, C)
        xa = rearrange(xa, "b (h w) c -> b h w c", h=int(H//self.pool_size), w=int(W//self.pool_size))
        # xa = tf.reshape(xa, [B, int(H//self.pool_size), int(W//self.pool_size), C])
        xa = self.uppool(xa)

        return xa

class ParallelConvolutionalTransformerMixingModule(tf.keras.layers.Layer):

    def __init__(self, dim, head, atten_head, pool_size, attn_drop=0., proj_drop=0., qkv_bias=False):
        super().__init__()
        self.dim = dim #32
        self.head = head #4
        self.atten_head = atten_head #1
        self.pool_size = pool_size #4
        self.head_dim = int(self.dim/self.head) #8
        self.global_dim = int(self.head_dim*self.atten_head) #8
        self.local_dim = self.dim-self.global_dim  #24
        self.local_layer = ConvolutionalBranch(dim=self.local_dim)
        self.global_layer = TransformerBranch(dim=self.global_dim, num_heads=atten_head, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, pool_size=pool_size)
        self.conv_fuse = tf.keras.layers.Conv2D(dim, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        self.tam = TransformerAttentionModule(self.global_dim)
        self.cam = ConvolutionAttentionModule(self.local_dim)

    def call(self, inputs):

        local_feature = inputs[:, :, :, :self.local_dim]
        local_feature = self.local_layer(local_feature)
        local_feature = self.cam(local_feature)

        global_feature = inputs[:, :, :, self.local_dim:]
        global_feature = self.global_layer(global_feature)
        global_feature = self.tam(global_feature)

        mix_feature = tf.keras.layers.Concatenate()([local_feature, global_feature])
        outputs = inputs+self.conv_fuse(mix_feature)
        outputs = self.proj_drop(outputs)

        return outputs

class Encoder(tf.keras.layers.Layer):
    def __init__(self, Minchnum, head, atten_head, pool_size, proj_drop, attn_drop=0., qkv_bias=False, depth=6,
                 use_learnable_mechanism=True):
        super().__init__()
        self.layer = []
        self.depth = depth
        self.learnable_variable = [
            tf.nn.sigmoid(tf.Variable(tf.random.truncated_normal([1], mean=0.0, stddev=0.05),
                                      dtype=tf.float32, trainable=True)) for _ in range (self.depth)]
        for i in range(depth):
            layer = ParallelConvolutionalTransformerMixingModule(Minchnum*2**i, head=head[i],
        atten_head=int(self.learnable_variable[i]*head[i]) if use_learnable_mechanism else atten_head[i], pool_size=pool_size[i],
                                    attn_drop=attn_drop, proj_drop=proj_drop[i])
            self.layer.append(layer)
        self.pool = [tf.keras.layers.MaxPool2D(pool_size=(2, 2)) for i in range(self.depth - 1)]
        self.proj = [ConvBnRelu(Minchnum * 2 ** (i+1)) for i in range(self.depth-1)]
    def call(self, x):
        outputs = []
        for i in range(self.depth):
            x = self.layer[i](x)
            outputs.append(x)
            if i != len(self.pool):
                x = self.pool[i](x)
                x = self.proj[i](x)
        return outputs

class Decoder(tf.keras.layers.Layer):
    def __init__(self, Minchnum, head, atten_head, pool_size,  proj_drop, attn_drop=0., qkv_bias=False, depth=6,
                 use_learnable_mechanism=True):
        super().__init__()
        self.depth = depth
        self.layer = []
        self.learnable_variable = [
            tf.nn.sigmoid(tf.Variable(tf.random.truncated_normal([1], mean=0.0, stddev=0.05),
                                      dtype=tf.float32, trainable=True)) for _ in range (self.depth)]
        for i in range(depth):
            layer = ParallelConvolutionalTransformerMixingModule(Minchnum * 2 ** (self.depth-1-i), head=head[self.depth-1-i],
                                    atten_head= int(head[self.depth-1-i]* self.learnable_variable[self.depth-1-i]) if use_learnable_mechanism
            else atten_head[self.depth-1-i], pool_size=pool_size[self.depth-1-i],
                                    attn_drop=attn_drop, proj_drop=proj_drop[self.depth-1-i])
            self.layer.append(layer)
        self.unpool = [DeConvBnRelu(Minchnum * 2 ** i) for i in range(self.depth-1)]
        self.proj = [tf.keras.layers.Dense(Minchnum * 2 ** i) for i in range(self.depth-1)]

    def call(self, x):
        outputs = []
        for i in range(self.depth):
            if i == 0:
                output = self.layer[i](x[-1])
            else:
                output = self.layer[i](outputs[-1])

            if i != len(self.unpool):
                output = self.unpool[self.depth-2-i](output)
                output = tf.keras.layers.Concatenate()([output, x[self.depth-2-i]])
                output = self.proj[self.depth-2-i](output)
            outputs.append(output)

        return outputs



class Mix_Graph_CrackNet(tf.keras.Model):
    def __init__(self, Minchnum=16, head=(4, 8, 16, 32, 64, 128), atten_head=(1, 3, 8, 24, 48, 96), pool_size=(8,4,2,1,1,1),
                 attn_drop=0., drop_path_rate=0.1, num_class=1, qkv_bias=False, depth=6, if_use_learnable_mechanism=True,
                 if_use_single_gbsc=True):
        super().__init__()
        self.if_use_single_gbsc = if_use_single_gbsc
        self.conv = tf.keras.layers.Conv2D(Minchnum, kernel_size=3, padding='same')
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]
        self.encoder = Encoder(Minchnum, head, atten_head, pool_size, dpr, attn_drop, qkv_bias, depth, if_use_learnable_mechanism)
        self.decoder = Decoder(Minchnum, head, atten_head, pool_size, dpr, attn_drop, qkv_bias, depth, if_use_learnable_mechanism)
        if if_use_single_gbsc:
            self.gbsc = GraphBasedSkipConnection(input_channels=Minchnum * 2 ** (depth - 1))
        else:
            self.gbsc = [GraphBasedSkipConnection(input_channels=Minchnum * 2 ** i) for i in range(depth)]
        self.result = tf.keras.layers.Conv2D(num_class, kernel_size=3, padding='same')
        if num_class == 1:
            self.Fin_out = tf.keras.layers.Activation('sigmoid')
        else:
            self.Fin_out = tf.keras.layers.Activation('softmax')

    def call(self, x):
        x = self.conv(x)
        encode_features = self.encoder(x)
        if self.if_use_single_gbsc:
            encode_features[-1] = self.gbsc(encode_features[-1])
        else:
            for i,graph_based_skip_connection in enumerate(self.gbsc):
                encode_features[i] = graph_based_skip_connection(encode_features[i])

        decode_features = self.decoder(encode_features)
        out = self.result(decode_features[-1])
        Fin_out = self.Fin_out(out)

        return Fin_out

if __name__ == "__main__":
    model = Mix_Graph_CrackNet(if_use_learnable_mechanism=True, Minchnum=32, if_use_single_gbsc=False)
    model.build((1, 256, 512, 1))
    model.summary()





