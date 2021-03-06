#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
#############################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer, relu_actfun, softplus_actfun, \
                      apply_mask
from InfNet import InfNet
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import to_fX

#
# Important symbolic variables:
#   Xd: Xd represents input at the "data variables" of the inferencer
#

class MultiStageModel(object):
    """
    Controller for training a multi-step iterative refinement model.

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: symbolic "data" input to this MultiStageModel
        x_out: symbolic "target" output for this MultiStageModel
        x_mask: symbolic binary "mask" describing known/missing target values
        p_s0_obs_given_z_obs: InfNet for s0 given z_obs
        p_hi_given_si: InfNet for hi given si
        p_sip1_given_si_hi: InfNet for sip1 given si and hi
        p_x_given_si_hi: InfNet for x given si and hi
        q_z_given_x: InfNet for z given x
        q_hi_given_x_si: InfNet for hi given x and si
        model_init_obs: whether to use a model-based initial obs state
        obs_dim: dimension of the observations to generate
        z_dim: dimension of the "initial" latent space
        h_dim: dimension of the "primary" latent space
        ir_steps: number of "iterative refinement" steps to perform
        params: REQUIRED PARAMS SHOWN BELOW
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None, x_in=None, \
            p_s0_obs_given_z_obs=None, p_hi_given_si=None, p_sip1_given_si_hi=None, \
            p_x_given_si_hi=None, q_z_given_x=None, q_hi_given_x_si=None, \
            obs_dim=None, z_dim=None, h_dim=None, \
            model_init_obs=True, ir_steps=2, \
            params=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # TODO: implement functionality for working with "latent" si
        assert(p_x_given_si_hi is None)

        # decide whether to initialize from a model or from a "constant"
        self.model_init_obs = model_init_obs

        # grab the user-provided parameters
        self.params = params
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))
        if 'obs_transform' in self.params:
            assert((self.params['obs_transform'] == 'sigmoid') or \
                    (self.params['obs_transform'] == 'none'))
            if self.params['obs_transform'] == 'sigmoid':
                self.obs_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.obs_transform = lambda x: x
        else:
            self.obs_transform = lambda x: T.nnet.sigmoid(x)
        if self.x_type == 'bernoulli':
            self.obs_transform = lambda x: T.nnet.sigmoid(x)

        # record the dimensions of various spaces relevant to this model
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.ir_steps = ir_steps

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x = x_in
        self.batch_reps = T.lscalar()

        # setup switching variable for changing between sampling/training
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.train_switch = theano.shared(value=zero_ary, name='msm_train_switch')
        self.set_train_switch(1.0)
        # setup a weight for pulling priors over hi given si towards a
        # shared global prior -- e.g. zero mean and unit variance.
        self.kzg_weight = theano.shared(value=zero_ary, name='msm_kzg_weight')
        self.set_kzg_weight(0.1)
        # this weight balances l1 vs. l2 penalty on posterior KLds
        self.l1l2_weight = theano.shared(value=zero_ary, name='msm_l1l2_weight')
        self.set_l1l2_weight(1.0)
        # this parameter controls dropout rate in the generator read function
        self.drop_rate = theano.shared(value=zero_ary, name='msm_drop_rate')
        self.set_drop_rate(0.0)

        #############################
        # Setup self.z and self.s0. #
        #############################
        print("Building MSM step 0...")
        obs_scale = 0.0
        if self.model_init_obs: # initialize obs state from generative model
            obs_scale = 1.0
        self.q_z_given_x = q_z_given_x.shared_param_clone(rng=rng, Xd=self.x)
        self.z = self.q_z_given_x.output
        self.p_s0_obs_given_z_obs = p_s0_obs_given_z_obs.shared_param_clone( \
                rng=rng, Xd=self.z)
        _s0_obs_model = self.p_s0_obs_given_z_obs.output_mean
        _s0_obs_const = self.p_s0_obs_given_z_obs.mu_layers[-1].b
        self.s0_obs = (obs_scale * _s0_obs_model) + \
                ((1.0 - obs_scale) * _s0_obs_const)
        self.output_logvar = self.p_s0_obs_given_z_obs.sigma_layers[-1].b
        self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.output_logvar)

        ###############################################################
        # Setup the iterative refinement loop, starting from self.s0. #
        ###############################################################
        self.p_hi_given_si = []       # holds p_hi_given_si for each i
        self.p_sip1_given_si_hi = []  # holds p_sip1_given_si_hi for each i
        self.q_hi_given_x_si = []     # holds q_hi_given_x_si for each i
        self.si = [self.s0_obs]       # holds si for each i
        self.hi = []                  # holds hi for each i
        for i in range(self.ir_steps):
            print("Building MSM step {0:d}...".format(i+1))
            si_obs = self.si[i]
            # get samples of next hi, conditioned on current si
            self.p_hi_given_si.append( \
                    p_hi_given_si.shared_param_clone(rng=rng, \
                    Xd=self.obs_transform(si_obs)))
            hi_p = self.p_hi_given_si[i].output
            # now we build the model for variational hi given si
            grad_ll = self.x - self.obs_transform(si_obs)
            self.q_hi_given_x_si.append(\
                    q_hi_given_x_si.shared_param_clone(rng=rng, \
                    Xd=T.horizontal_stack( \
                    grad_ll, self.obs_transform(si_obs))))
            hi_q = self.q_hi_given_x_si[i].output
            # make hi samples that can be switched between hi_p and hi_q
            self.hi.append( ((self.train_switch[0] * hi_q) + \
                    ((1.0 - self.train_switch[0]) * hi_p)) )
            # p_sip1_given_si_hi is conditioned on hi.
            self.p_sip1_given_si_hi.append( \
                    p_sip1_given_si_hi.shared_param_clone(rng=rng, \
                    Xd=self.hi[i]))
            # construct the update from si_obs to sip1_obs
            sip1_obs = si_obs + self.p_sip1_given_si_hi[i].output_mean
            # record the updated state of the generative process
            self.si.append(sip1_obs)

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        self.lr_1 = theano.shared(value=zero_ary, name='msm_lr_1')
        self.lr_2 = theano.shared(value=zero_ary, name='msm_lr_2')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='msm_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='msm_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='msm_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_1 = theano.shared(value=zero_ary, name='msm_lam_kld_1')
        self.lam_kld_2 = theano.shared(value=zero_ary, name='msm_lam_kld_2')
        self.set_lam_kld(lam_kld_1=1.0, lam_kld_2=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in "group 1"
        self.group_1_params = []
        self.group_1_params.extend(self.q_z_given_x.mlp_params)
        self.group_1_params.extend(self.p_s0_obs_given_z_obs.mlp_params)
        # Grab all of the "optimizable" parameters in "group 2"
        self.group_2_params = []
        for i in range(self.ir_steps):
            self.group_2_params.extend(self.q_hi_given_x_si[i].mlp_params)
            self.group_2_params.extend(self.p_hi_given_si[i].mlp_params)
            self.group_2_params.extend(self.p_sip1_given_si_hi[i].mlp_params)
        # Make a joint list of parameters group 1/2
        self.joint_params = self.group_1_params + self.group_2_params

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_z, self.kld_hi_cond, self.kld_hi_glob = \
                self._construct_kld_costs()
        self.kld_cost = (self.lam_kld_1[0] * T.mean(self.kld_z)) + \
                (self.lam_kld_2[0] * (T.mean(self.kld_hi_cond) + \
                (self.kzg_weight[0] * T.mean(self.kld_hi_glob))))
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self._construct_nll_costs()
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.group_1_updates = get_adam_updates(params=self.group_1_params, \
                grads=self.joint_grads, alpha=self.lr_1, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.group_2_updates = get_adam_updates(params=self.group_2_params, \
                grads=self.joint_grads, alpha=self.lr_2, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        self.joint_updates = OrderedDict()
        for k in self.group_1_updates:
            self.joint_updates[k] = self.group_1_updates[k]
        for k in self.group_2_updates:
            self.joint_updates[k] = self.group_2_updates[k]

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        self.compute_post_klds = self._construct_compute_post_klds()
        self.compute_fe_terms = self._construct_compute_fe_terms()
        self.sample_from_prior = self._construct_sample_from_prior()
        # make easy access points for some interesting parameters
        self.inf_1_weights = self.q_z_given_x.shared_layers[0].W
        self.gen_1_weights = self.p_s0_obs_given_z_obs.mu_layers[-1].W
        self.inf_2_weights = self.q_hi_given_x_si[0].shared_layers[0].W
        self.gen_2_weights = self.p_sip1_given_si_hi[0].mu_layers[-1].W
        self.gen_inf_weights = self.p_hi_given_si[0].shared_layers[0].W
        return

    def set_sgd_params(self, lr_1=0.01, lr_2=0.01, \
                mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates
        new_lr_1 = zero_ary + lr_1
        self.lr_1.set_value(new_lr_1.astype(theano.config.floatX))
        new_lr_2 = zero_ary + lr_2
        self.lr_2.set_value(new_lr_2.astype(theano.config.floatX))
        # set momentums
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_kld(self, lam_kld_1=1.0, lam_kld_2=1.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_1
        self.lam_kld_1.set_value(new_lam.astype(theano.config.floatX))
        new_lam = zero_ary + lam_kld_2
        self.lam_kld_2.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_train_switch(self, switch_val=0.0):
        """
        Set the switch for changing between training and sampling behavior.
        """
        if (switch_val < 0.5):
            switch_val = 0.0
        else:
            switch_val = 1.0
        zero_ary = np.zeros((1,))
        new_val = zero_ary + switch_val
        new_val = new_val.astype(theano.config.floatX)
        self.train_switch.set_value(new_val)
        return

    def set_kzg_weight(self, kzg_weight=0.2):
        """
        Set the weight for shaping penalty on conditional priors over zt.
        """
        assert(kzg_weight >= 0.0)
        zero_ary = np.zeros((1,))
        new_val = zero_ary + kzg_weight
        new_val = new_val.astype(theano.config.floatX)
        self.kzg_weight.set_value(new_val)
        return

    def set_l1l2_weight(self, l1l2_weight=1.0):
        """
        Set the weight for shaping penalty on posterior KLds.
        """
        assert((l1l2_weight >= 0.0) and (l1l2_weight <= 1.0))
        zero_ary = np.zeros((1,))
        new_val = zero_ary + l1l2_weight
        new_val = new_val.astype(theano.config.floatX)
        self.l1l2_weight.set_value(new_val)
        return

    def set_drop_rate(self, drop_rate=0.0):
        """
        Set the dropout rate for generator read function.
        """
        assert((drop_rate >= 0.0) and (drop_rate <= 1.0))
        zero_ary = np.zeros((1,))
        new_val = zero_ary + drop_rate
        new_val = new_val.astype(theano.config.floatX)
        self.drop_rate.set_value(new_val)
        return

    def set_input_bias(self, new_bias=None):
        """
        Set the output layer bias.
        """
        new_bias = new_bias.astype(theano.config.floatX)
        self.q_z_given_x.shared_layers[0].b_in.set_value(new_bias)
        return

    def set_obs_bias(self, new_obs_bias=None):
        """
        Set initial bias on the obs part of state.
        """
        assert(new_obs_bias.shape[0] == self.obs_dim)
        new_bias = np.zeros((self.obs_dim,)) + new_obs_bias
        new_bias = new_bias.astype(theano.config.floatX)
        self.p_s0_obs_given_z_obs.mu_layers[-1].b.set_value(new_bias)
        return

    def _construct_nll_costs(self):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self.obs_transform(self.si[-1])
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(self.x, xh)
        else:
            ll_costs = log_prob_gaussian2(self.x, xh, \
                    log_vars=self.bounded_logvar)
        nll_costs = -ll_costs
        return nll_costs

    def _construct_kld_costs(self):
        """
        Construct the posterior KL-divergence part of cost to minimize.
        """
        # construct KLd cost for the distributions over hi. the prior over
        # hi is given by a distribution conditioned on si, which we estimate
        # using self.p_hi_given_si[i]. the conditionals produced by each
        # self.p_hi_given_si[i] will also be regularized towards a shared
        # prior, e.g. a Gaussian with zero mean and unit variance.
        kld_hi_conds = []
        kld_hi_globs = []
        for i in range(self.ir_steps):
            kld_hi_cond = gaussian_kld( \
                    self.q_hi_given_x_si[i].output_mean, \
                    self.q_hi_given_x_si[i].output_logvar, \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar)
            kld_hi_glob = gaussian_kld( \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar, \
                    0.0, 0.0)
            kld_hi_cond_l1l2 = (self.l1l2_weight[0] * kld_hi_cond) + \
                    ((1.0 - self.l1l2_weight[0]) * kld_hi_cond**2.0)
            kld_hi_conds.append(T.sum(kld_hi_cond_l1l2, \
                    axis=1, keepdims=True))
            kld_hi_globs.append(T.sum(kld_hi_glob**2.0, \
                    axis=1, keepdims=True))
        # compute the batch-wise costs
        kld_hi_cond = sum(kld_hi_conds)
        kld_hi_glob = sum(kld_hi_globs)
        # construct KLd cost for the distributions over z
        kld_z_all = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                0.0, 0.0)
        kld_z_l1l2 = (self.l1l2_weight[0] * kld_z_all) + \
                ((1.0 - self.l1l2_weight[0]) * kld_z_all**2.0)
        kld_z = T.sum(kld_z_l1l2, \
                axis=1, keepdims=True)
        return [kld_z, kld_hi_cond, kld_hi_glob]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        x = T.matrix()
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_cost, self.kld_cost, \
                self.reg_cost]
        # compile the theano function
        func = theano.function(inputs=[ x, self.batch_reps ], \
                outputs=outputs, \
                givens={ self.x: x.repeat(self.batch_reps, axis=0) }, \
                updates=self.joint_updates)
        return func

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        x_in = T.matrix()
        # construct values to output
        nll = self._construct_nll_costs()
        kld = self.kld_z + self.kld_hi_cond
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[x_in], \
                outputs=[nll, kld], givens={self.x: x_in})
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(X, sample_count):
            nll_sum = np.zeros((X.shape[0],))
            kld_sum = np.zeros((X.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(X)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_compute_post_klds(self):
        """
        Construct theano function to compute the info about the variational
        approximate posteriors for some inputs.
        """
        # setup some symbolic variables for theano to deal with
        x = T.matrix()
        # construct symbolic expressions for the desired KLds
        cond_klds = []
        glob_klds = []
        for i in range(self.ir_steps):
            kld_hi_cond = gaussian_kld(self.q_hi_given_x_si[i].output_mean, \
                    self.q_hi_given_x_si[i].output_logvar, \
                    self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar)
            kld_hi_glob = gaussian_kld(self.p_hi_given_si[i].output_mean, \
                    self.p_hi_given_si[i].output_logvar, 0.0, 0.0)
            cond_klds.append(kld_hi_cond)
            glob_klds.append(kld_hi_glob)
        # gather conditional and global klds for all IR steps
        all_klds = cond_klds + glob_klds
        # gather kld for the initialization step
        kld_z_all = gaussian_kld(self.q_z_given_x.output_mean, \
                self.q_z_given_x.output_logvar, \
                0.0, 0.0)
        all_klds.append(kld_z_all)
        # compile theano function for a one-sample free-energy estimate
        kld_func = theano.function(inputs=[x], outputs=all_klds, \
                givens={ self.x: x })
        def post_kld_computer(X):
            f_all_klds = kld_func(X)
            f_kld_z = f_all_klds[-1]
            f_kld_hi_cond = np.zeros(f_all_klds[0].shape)
            f_kld_hi_glob = np.zeros(f_all_klds[0].shape)
            for j in range(self.ir_steps):
                f_kld_hi_cond += f_all_klds[j]
                f_kld_hi_glob += f_all_klds[j + self.ir_steps]
            return [f_kld_z, f_kld_hi_cond, f_kld_hi_glob]
        return post_kld_computer

    def _construct_sample_from_prior(self):
        """
        Construct a function for drawing independent samples from the
        distribution generated by this MultiStageModel. This function returns
        the full sequence of "partially completed" examples.
        """
        z_sym = T.matrix()
        x_sym = T.matrix()
        oputs = [self.obs_transform(s) for s in self.si]
        sample_func = theano.function(inputs=[z_sym, x_sym], outputs=oputs, \
                givens={ self.z: z_sym, \
                        self.x: T.zeros_like(x_sym) })
        def prior_sampler(samp_count):
            x_samps = np.zeros((samp_count, self.obs_dim))
            x_samps = x_samps.astype(theano.config.floatX)
            old_switch = self.train_switch.get_value(borrow=False)
            # set model to generation mode
            self.set_train_switch(switch_val=0.0)
            z_samps = npr.randn(samp_count, self.z_dim)
            z_samps = z_samps.astype(theano.config.floatX)
            model_samps = sample_func(z_samps, x_samps)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            return model_samps
        return prior_sampler

if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
