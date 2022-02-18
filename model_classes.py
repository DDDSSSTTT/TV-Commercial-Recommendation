import tensorflow as tf
class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb, itemIDs, userIDs):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)], stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)], stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs), K], stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs), K], stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] + \
            tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU ** 2) + \
                            tf.reduce_sum(self.betaI ** 2) + \
                            tf.reduce_sum(self.gammaU ** 2) + \
                            tf.reduce_sum(self.gammaI ** 2))

    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i + \
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred

    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)

# Add some features of previous ratio (linear)
class SPLFM(tf.keras.Model):
    def __init__(self, mu, K, lamb, itemIDs, userIDs, linear = 1, fIDs =[]):
        super(SPLFM, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)], stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)], stddev=0.001))
        if not linear:
            self.thetaF = tf.Variable(tf.random.normal([len(fIDs),K], stddev=0.001))
            self.thetaU = tf.Variable(tf.random.normal([len(userIDs),K], stddev=0.001))
        else:
            self.thetaU = tf.Variable(tf.random.normal([len(userIDs)], stddev=0.001))

        self.gammaU = tf.Variable(tf.random.normal([len(userIDs), K], stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs), K], stddev=0.001))
        self.lamb = lamb
        self.linear = linear

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i, f):
        if self.linear:
            p = self.alpha + self.betaU[u] + self.betaI[i] + \
                tf.tensordot(self.thetaU[u], f, 1) + \
                tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        else:
            p = self.alpha + self.betaU[u] + self.betaI[i] + \
                tf.tensordot(self.thetaU[u],self.thetaF[f],1) + \
                tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU ** 2) + \
                            tf.reduce_sum(self.betaI ** 2) + \
                            tf.reduce_sum(self.thetaU ** 2) + \
                            tf.reduce_sum(self.gammaU ** 2) + \
                            tf.reduce_sum(self.gammaI ** 2))

    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI, sampleF):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        if self.linear:
            f = tf.convert_to_tensor(sampleF, dtype=tf.float32)
        else:
            f = tf.convert_to_tensor(sampleF, dtype=tf.int32)
            theta_f = tf.nn.embedding_lookup(self.thetaF, f)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        theta_u = tf.nn.embedding_lookup(self.thetaU, u)

        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        if self.linear:
            pred = self.alpha + beta_u + beta_i + \
                theta_u * f + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        else:
            pred = self.alpha + beta_u + beta_i + \
                tf.reduce_sum(tf.multiply(theta_u, theta_f), 1) + \
                   tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred

    # Loss
    def call(self, sampleU, sampleI, sampleF, sampleR):
        pred = self.predictSample(sampleU, sampleI, sampleF)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)


class FPMC(tf.keras.Model):
    def __init__(self, mu, K, lamb, UI = 1, IJ = 1, userIDs=[], itemIDs=[]):
        super(FPMC, self).__init__()
        # Initialize variables
        self.alpha = tf.Variable(mu)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaUI = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaIU = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIJ = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaJI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb
        # Which terms to include
        self.UI = UI
        self.IJ = IJ

    # Prediction for a single instance
    def predict(self, u, i, j):
        p = self.alpha + self.UI*self.betaU[u] + self.betaI[i] + self.UI * tf.tensordot(self.gammaUI[u], self.gammaIU[i], 1) +\
                            self.IJ * tf.tensordot(self.gammaIJ[i], self.gammaJI[j], 1)
        return p
    #Prediction for a tensor
    def predictSample(self,sampleU,sampleI,sampleJ):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        j = tf.convert_to_tensor(sampleJ, dtype=tf.int32)
        gamma_ui = tf.nn.embedding_lookup(self.gammaUI, u)
        gamma_iu = tf.nn.embedding_lookup(self.gammaIU, i)
        gamma_ij = tf.nn.embedding_lookup(self.gammaIJ, i)
        gamma_ji = tf.nn.embedding_lookup(self.gammaJI, j)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        beta_u = tf.nn.embedding_lookup(self.betaU,u)
        little_bias = beta_i + self.UI*beta_u + self.UI*tf.reduce_sum(tf.multiply(gamma_ui, gamma_iu,1),1) + \
                      self.IJ*tf.reduce_sum(tf.multiply(gamma_ij,gamma_ji,1),1)
        pred =self.alpha + little_bias
        return pred

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.betaU) +\
                            tf.nn.l2_loss(self.gammaUI) +\
                            tf.nn.l2_loss(self.gammaIU) +\
                            tf.nn.l2_loss(self.gammaIJ) +\
                            tf.nn.l2_loss(self.gammaJI))

    def call(self, sampleU, # user
                   sampleI, # item
                   sampleJ,# previous item
                   sampleR): # rating

        pred = self.predictSample(sampleU, sampleI, sampleJ)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)

class FPMCWT(tf.keras.Model):
    def __init__(self, mu, K, lamb, UI = 1, IJ = 1, timeIDs=[], userIDs=[], itemIDs=[]):
        super(FPMCWT, self).__init__()
        # Initialize variables
        self.alpha = tf.Variable(mu)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.betaT = tf.Variable(tf.random.normal([len(timeIDs)],stddev=0.001))
        self.gammaUI = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaIU = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIJ = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaJI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIT = tf.Variable(tf.random.normal([len(timeIDs),K],stddev=0.001))
        self.gammaTI = tf.Variable(tf.random.normal([len(timeIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb
        # Which terms to include
        self.UI = UI
        self.IJ = IJ

    # Prediction for a single instance
    def predict(self, t, u, i, j):
        p = self.alpha + self.UI*self.betaU[u] + self.betaI[i] +self.betaT[t] + self.UI * tf.tensordot(self.gammaUI[u], self.gammaIU[i], 1) +\
                            self.IJ * tf.tensordot(self.gammaIJ[i], self.gammaJI[j], 1) + tf.tensordot(self.gammaTI[t], self.gammeIT[i],1)
        return p
    #Prediction for a tensor
    def predictSample(self, sampleT, sampleU,sampleI,sampleJ):
        t = tf.convert_to_tensor(sampleT, dtype=tf.int32)
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        j = tf.convert_to_tensor(sampleJ, dtype=tf.int32)
        gamma_ui = tf.nn.embedding_lookup(self.gammaUI, u)
        gamma_iu = tf.nn.embedding_lookup(self.gammaIU, i)
        gamma_ij = tf.nn.embedding_lookup(self.gammaIJ, i)
        gamma_ji = tf.nn.embedding_lookup(self.gammaJI, j)
        gamma_ti = tf.nn.embedding_lookup(self.gammaTI,t)
        gamma_it = tf.nn.embedding_lookup(self.gammaIT,i)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_t = tf.nn.embedding_lookup(self.betaT, t)
        little_bias = beta_i + beta_t + self.UI*beta_u + self.UI*tf.reduce_sum(tf.multiply(gamma_ui, gamma_iu,1),1) + \
                      self.IJ*tf.reduce_sum(tf.multiply(gamma_ij,gamma_ji,1),1) + tf.reduce_sum(tf.multiply(gamma_ti,gamma_it,1),1)
        pred =self.alpha + little_bias
        return pred

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.betaU) +\
                            tf.nn.l2_loss(self.betaT) +\
                            tf.nn.l2_loss(self.gammaUI) +\
                            tf.nn.l2_loss(self.gammaIU) + \
                            tf.nn.l2_loss(self.gammaTI) + \
                            tf.nn.l2_loss(self.gammaIT) + \
                            tf.nn.l2_loss(self.gammaIJ) +\
                            tf.nn.l2_loss(self.gammaJI))


    def call(self, sampleT, #time
                   sampleU, # user
                   sampleI, # item
                   sampleJ,# previous item
                   sampleR): # rating

        pred = self.predictSample(sampleT, sampleU, sampleI, sampleJ)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)


class RBMCWT(tf.keras.Model):
    def __init__(self, mu, K, lamb, UI = 1, IJ = 1, timeIDs=[], userIDs=[], itemIDs=[]):
        super(RBMCWT, self).__init__()
        # Initialize variables
        self.alpha = tf.Variable(mu)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.betaT = tf.Variable(tf.random.normal([len(timeIDs)],stddev=0.001))
        self.thetaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.gammaUI = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaIU = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIJ = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaJI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIT = tf.Variable(tf.random.normal([len(timeIDs),K],stddev=0.001))
        self.gammaTI = tf.Variable(tf.random.normal([len(timeIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb
        # Which terms to include
        self.UI = UI
        self.IJ = IJ

    # Prediction for a single instance
    def predict(self, t, u, i, j, pr):
        p = self.alpha + self.thetaU[u] * pr + \
            self.UI*self.betaU[u] + self.betaI[i] +self.betaT[t] + self.UI * tf.tensordot(self.gammaUI[u], self.gammaIU[i], 1) +\
                            self.IJ * tf.tensordot(self.gammaIJ[i], self.gammaJI[j], 1) + tf.tensordot(self.gammaTI[t], self.gammeIT[i],1)
        return p
    #Prediction for a tensor
    def predictSample(self, sampleT, sampleU,sampleI,sampleJ, samplePR):
        t = tf.convert_to_tensor(sampleT, dtype=tf.int32)
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        j = tf.convert_to_tensor(sampleJ, dtype=tf.int32)
        pr = tf.convert_to_tensor(samplePR, dtype=tf.float32)
        gamma_ui = tf.nn.embedding_lookup(self.gammaUI, u)
        gamma_iu = tf.nn.embedding_lookup(self.gammaIU, i)
        gamma_ij = tf.nn.embedding_lookup(self.gammaIJ, i)
        gamma_ji = tf.nn.embedding_lookup(self.gammaJI, j)
        gamma_ti = tf.nn.embedding_lookup(self.gammaTI,t)
        gamma_it = tf.nn.embedding_lookup(self.gammaIT,i)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_t = tf.nn.embedding_lookup(self.betaT, t)
        theta_u = tf.nn.embedding_lookup(self.thetaU, u)
        little_bias = beta_i + beta_t + theta_u * pr +\
                      self.UI*beta_u + self.UI*tf.reduce_sum(tf.multiply(gamma_ui, gamma_iu,1),1) + \
                      self.IJ*tf.reduce_sum(tf.multiply(gamma_ij,gamma_ji,1),1) + tf.reduce_sum(tf.multiply(gamma_ti,gamma_it,1),1)
        pred =self.alpha + little_bias
        return pred

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.betaU) +\
                            tf.nn.l2_loss(self.betaT) +\
                            tf.nn.l2_loss(self.thetaU) +\
                            tf.nn.l2_loss(self.gammaUI) +\
                            tf.nn.l2_loss(self.gammaIU) + \
                            tf.nn.l2_loss(self.gammaTI) + \
                            tf.nn.l2_loss(self.gammaIT) + \
                            tf.nn.l2_loss(self.gammaIJ) +\
                            tf.nn.l2_loss(self.gammaJI))


    def call(self, sampleT, #time
                   sampleU, # user
                   sampleI, # item
                   sampleJ,# previous item
                   samplePR,
                   sampleR): # rating

        pred = self.predictSample(sampleT, sampleU, sampleI, sampleJ, samplePR)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)

class RBMC(tf.keras.Model):
    def __init__(self, mu, K, lamb, UI = 1, IJ = 1,userIDs=[], itemIDs=[]):
        super(RBMC, self).__init__()
        # Initialize variables
        self.alpha = tf.Variable(mu)
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.thetaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.gammaUI = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaIU = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaIJ = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        self.gammaJI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb
        # Which terms to include
        self.UI = UI
        self.IJ = IJ

    # Prediction for a single instance
    def predict(self, u, i, j, pr):
        p = self.alpha + self.thetaU[u] * pr + \
            self.UI*self.betaU[u] + self.betaI[i]+ self.UI * tf.tensordot(self.gammaUI[u], self.gammaIU[i], 1) +\
                            self.IJ * tf.tensordot(self.gammaIJ[i], self.gammaJI[j], 1)
        return p
    #Prediction for a tensor
    def predictSample(self, sampleU,sampleI,sampleJ, samplePR):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        j = tf.convert_to_tensor(sampleJ, dtype=tf.int32)
        pr = tf.convert_to_tensor(samplePR, dtype=tf.float32)
        gamma_ui = tf.nn.embedding_lookup(self.gammaUI, u)
        gamma_iu = tf.nn.embedding_lookup(self.gammaIU, i)
        gamma_ij = tf.nn.embedding_lookup(self.gammaIJ, i)
        gamma_ji = tf.nn.embedding_lookup(self.gammaJI, j)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        theta_u = tf.nn.embedding_lookup(self.thetaU, u)
        little_bias = beta_i + theta_u * pr +\
                      self.UI*beta_u + self.UI*tf.reduce_sum(tf.multiply(gamma_ui, gamma_iu,1),1) + \
                      self.IJ*tf.reduce_sum(tf.multiply(gamma_ij,gamma_ji,1),1)
        pred =self.alpha + little_bias
        return pred

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.betaU) +\
                            tf.nn.l2_loss(self.thetaU) +\
                            tf.nn.l2_loss(self.gammaUI) +\
                            tf.nn.l2_loss(self.gammaIU) + \
                            tf.nn.l2_loss(self.gammaIJ) +\
                            tf.nn.l2_loss(self.gammaJI))


    def call(self, sampleU, # user
                   sampleI, # item
                   sampleJ,# previous item
                   samplePR,
                   sampleR): # rating

        pred = self.predictSample(sampleU, sampleI, sampleJ, samplePR)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)