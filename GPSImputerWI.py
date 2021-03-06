#############################################################################
# Code for managing and training a variational Iterative Refinement Model.  #
#############################################################################

# basic python
import cPickle
import numpy as np
import numpy.random as npr
from collections import OrderedDict
import numexpr as ne

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from DKCode import get_adam_updates, get_adadelta_updates
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import to_fX

#######################################
# IMPLEMENT THE THING THAT DOES STUFF #
#######################################

class GPSImputerWI(object):
    """
    Controller for training a multi-step imputater via guided policy search.

    This model adds an "initialization" step, prior to iterative refinement.
    The init step requires three additional networks: action selectors for both
    the primary and guide policies, and an action->state transformer.


    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        x_in: the initial state for imputation
        x_out: the goal state for imputation
        x_mask: mask for state dims to keep fixed during imputation
        p_h_given_x: InfNet for stochastic part of init step
        p_s0_given_h: HydraNet for deterministic part of init step
        p_zi_given_xi: InfNet for stochastic part of refinement steps
        p_sip1_given_zi: HydraNet for deterministic part of refinement steps
        p_x_given_si: HydraNet for transform from s-space to x-space
        q_h_given_x: InfNet for the guide policy (init step)
        q_zi_given_xi: InfNet for the guide policy (refinement steps)
        params: REQUIRED PARAMS SHOWN BELOW
                x_dim: dimension of inputs to reconstruct
                h_dim: dimension of latent space for init step
                z_dim: dimension of latent space for policy wobble
                s_dim: dimension of space for hypothesis construction
                use_p_x_given_si: boolean for whether to use ----
                imp_steps: number of reconstruction steps to perform
                step_type: either "add" or "jump"
                x_type: can be "bernoulli" or "gaussian"
                obs_transform: can be 'none' or 'sigmoid'
    """
    def __init__(self, rng=None,
            x_in=None, x_mask=None, x_out=None, \
            p_h_given_x=None, \
            p_s0_given_h=None, \
            p_zi_given_xi=None, \
            p_sip1_given_zi=None, \
            p_x_given_si=None, \
            q_h_given_x=None, \
            q_zi_given_xi=None, \
            params=None, \
            shared_param_dicts=None):
        # setup a rng for this GIPair
        self.rng = RandStream(rng.randint(100000))

        # grab the user-provided parameters
        self.params = params
        self.x_dim = self.params['x_dim']
        self.h_dim = self.params['h_dim']
        self.z_dim = self.params['z_dim']
        self.s_dim = self.params['s_dim']
        self.use_p_x_given_si = self.params['use_p_x_given_si']
        self.imp_steps = self.params['imp_steps']
        self.step_type = self.params['step_type']
        self.x_type = self.params['x_type']
        if self.use_p_x_given_si:
            print("Constructing hypotheses via p_x_given_si...")
        else:
            print("Constructing hypotheses directly in x-space...")
            assert(self.s_dim == self.x_dim)
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
        self.shared_param_dicts = shared_param_dicts

        assert((self.step_type == 'add') or (self.step_type == 'jump'))

        # grab handles to the relevant InfNets
        self.p_h_given_x = p_h_given_x
        self.p_s0_given_h = p_s0_given_h
        self.p_zi_given_xi = p_zi_given_xi
        self.p_sip1_given_zi = p_sip1_given_zi
        self.p_x_given_si = p_x_given_si
        self.q_h_given_x = q_h_given_x
        self.q_zi_given_xi = q_zi_given_xi

        # record the symbolic variables that will provide inputs to the
        # computation graph created to describe this MultiStageModel
        self.x_in = x_in
        self.x_out = x_out
        self.x_mask = x_mask
        self.zi_zmuv = T.tensor3()

        # setup switching variable for changing between sampling/training
        zero_ary = to_fX( np.zeros((1,)) )
        self.train_switch = theano.shared(value=zero_ary, name='gpsi_train_switch')
        self.set_train_switch(1.0)

        if self.shared_param_dicts is None:
            # initialize parameters "owned" by this model
            init_ary = to_fX( np.zeros((self.x_dim,)) )
            self.s_null = theano.shared(value=init_ary, name='gpis_sn')
            self.grad_null = theano.shared(value=init_ary, name='gpsi_gn')
            self.obs_logvar = theano.shared(value=zero_ary, name='gpsi_obs_logvar')
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.shared_param_dicts = {}
            self.shared_param_dicts['s_null'] = self.s_null
            self.shared_param_dicts['grad_null'] = self.grad_null
            self.shared_param_dicts['obs_logvar'] = self.obs_logvar
            self.x_null = self._from_si_to_x(self.s_null)
        else:
            # grab the parameters required by this model from a given dict
            self.s_null = self.shared_param_dicts['s_null']
            self.grad_null = self.shared_param_dicts['grad_null']
            self.obs_logvar = self.shared_param_dicts['obs_logvar']
            self.bounded_logvar = 8.0 * T.tanh((1.0/8.0) * self.obs_logvar[0])
            self.x_null = self._from_si_to_x(self.s_null)

        ##############################################
        # Compute results of the initialization step #
        ##############################################
        self.x_init = (self.x_mask * self.x_in) + \
                      ((1.0 - self.x_mask) * self.x_null)
        # sample from primary and guide conditionals over h
        h_p_mean, h_p_logvar, h_p = \
                self.p_h_given_x.apply(self.x_init, do_samples=True)
        h_q_mean, h_q_logvar, h_q = \
                self.q_h_given_x.apply(self.x_in, do_samples=True)
        # make h samples that can be switched between h_p and h_q
        self.h = ((self.train_switch[0] * h_q) + \
                 ((1.0 - self.train_switch[0]) * h_p))
        # get the emitted initial state s0 (sampled via either p or q)
        hydra_out = self.p_s0_given_h.apply(self.h)
        self.s0 = hydra_out[0]
        # compute NLL reconstruction cost for the initialization step
        self.nll0 = self._construct_nll_costs(self.s0, self.x_out, self.x_mask)
        # compute KLds for the initialization step
        self.kldh_q2p = gaussian_kld(h_q_mean, h_q_logvar, \
                                     h_p_mean, h_p_logvar) # KL(q || p)
        self.kldh_p2q = gaussian_kld(h_p_mean, h_p_logvar, \
                                     h_q_mean, h_q_logvar) # KL(p || q)
        self.kldh_p2g = gaussian_kld(h_p_mean, h_p_logvar, \
                                     0.0, 0.0) # KL(p || global prior)

        ##################################################
        # Setup the iterative imputation loop using scan #
        ##################################################
        self.ones_mask = T.ones_like(self.x_mask)
        def imp_step_func(zi_zmuv, si):
            si_as_x = self._from_si_to_x(si)
            xi_unmasked = self.x_out
            xi_masked = (self.x_mask * xi_unmasked) + \
                        ((1.0 - self.x_mask) * si_as_x)
            grad_unmasked = self.x_out - si_as_x
            grad_masked = (self.x_mask * grad_unmasked) + \
                          ((1.0 - self.x_mask) * self.grad_null)
            # get samples of next zi, according to the global policy
            zi_p_mean, zi_p_logvar = self.p_zi_given_xi.apply( \
                    T.horizontal_stack(xi_masked, grad_masked), \
                    do_samples=False)
            zi_p = zi_p_mean + (T.exp(0.5 * zi_p_logvar) * zi_zmuv)
            # get samples of next zi, according to the guide policy
            zi_q_mean, zi_q_logvar = self.q_zi_given_xi.apply( \
                    T.horizontal_stack(xi_masked, grad_unmasked), \
                    do_samples=False)
            zi_q = zi_q_mean + (T.exp(0.5 * zi_q_logvar) * zi_zmuv)

            # make zi samples that can be switched between zi_p and zi_q
            zi = ((self.train_switch[0] * zi_q) + \
                 ((1.0 - self.train_switch[0]) * zi_p))
            # compute relevant KLds for this step
            kldi_q2p = gaussian_kld(zi_q_mean, zi_q_logvar, \
                                    zi_p_mean, zi_p_logvar) # KL(q || p)
            kldi_p2q = gaussian_kld(zi_p_mean, zi_p_logvar, \
                                    zi_q_mean, zi_q_logvar) # KL(p || q)
            kldi_p2g = gaussian_kld(zi_p_mean, zi_p_logvar, \
                                    0.0, 0.0) # KL(p || global prior)

            # compute the next si, given the sampled zi
            hydra_out = self.p_sip1_given_zi.apply(zi)
            si_step = hydra_out[0]
            if (self.step_type == 'jump'):
                # jump steps always completely overwrite the current guesses
                sip1 = si_step
            else:
                # additive steps update the current guesses like an LSTM
                write_gate = T.nnet.sigmoid(3.0 + hydra_out[1])
                erase_gate = T.nnet.sigmoid(3.0 + hydra_out[2])
                sip1 = (erase_gate * si) + (write_gate * si_step)
            # compute NLL for the current imputation
            nlli = self._construct_nll_costs(sip1, self.x_out, self.x_mask)
            return sip1, nlli, kldi_q2p, kldi_p2q, kldi_p2g

        # apply scan op for the sequential imputation loop
        init_vals = [self.s0, None, None, None, None]
        self.scan_results, self.scan_updates = theano.scan(imp_step_func, \
                    outputs_info=init_vals, sequences=self.zi_zmuv)

        self.si = self.scan_results[0]
        self.nlli = self.scan_results[1]
        self.kldi_q2p = self.scan_results[2]
        self.kldi_p2q = self.scan_results[3]
        self.kldi_p2g = self.scan_results[4]

        ######################################################################
        # ALL SYMBOLIC VARS NEEDED FOR THE OBJECTIVE SHOULD NOW BE AVAILABLE #
        ######################################################################

        # shared var learning rate for generator and inferencer
        zero_ary = to_fX( np.zeros((1,)) )
        self.lr = theano.shared(value=zero_ary, name='gpsi_lr')
        # shared var momentum parameters for generator and inferencer
        self.mom_1 = theano.shared(value=zero_ary, name='gpsi_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='gpsi_mom_2')
        # init parameters for controlling learning dynamics
        self.set_sgd_params()
        # init shared var for weighting nll of data given posterior sample
        self.lam_nll = theano.shared(value=zero_ary, name='gpsi_lam_nll')
        self.set_lam_nll(lam_nll=1.0)
        # init shared var for weighting prior kld against reconstruction
        self.lam_kld_p = theano.shared(value=zero_ary, name='gpsi_lam_kld_p')
        self.lam_kld_q = theano.shared(value=zero_ary, name='gpsi_lam_kld_q')
        self.lam_kld_g = theano.shared(value=zero_ary, name='gpsi_lam_kld_g')
        self.lam_kld_s = theano.shared(value=zero_ary, name='gpsi_lam_kld_s')
        self.set_lam_kld(lam_kld_p=0.0, lam_kld_q=1.0, lam_kld_g=0.0, lam_kld_s=0.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='msm_lam_l2w')
        self.set_lam_l2w(1e-5)

        # Grab all of the "optimizable" parameters in the model
        self.joint_params = [self.s_null, self.grad_null, self.obs_logvar]
        self.joint_params.extend(self.p_zi_given_xi.mlp_params)
        self.joint_params.extend(self.p_sip1_given_zi.mlp_params)
        self.joint_params.extend(self.p_x_given_si.mlp_params)
        self.joint_params.extend(self.q_zi_given_xi.mlp_params)

        #################################
        # CONSTRUCT THE KLD-BASED COSTS #
        #################################
        self.kld_p, self.kld_q, self.kld_g, self.kld_s = \
                self._construct_kld_costs(p=1.0)
        self.kld_costs = (self.lam_kld_p[0] * self.kld_p) + \
                         (self.lam_kld_q[0] * self.kld_q) + \
                         (self.lam_kld_g[0] * self.kld_g) + \
                         (self.lam_kld_s[0] * self.kld_s)
        self.kld_cost = T.mean(self.kld_costs)
        #################################
        # CONSTRUCT THE NLL-BASED COSTS #
        #################################
        self.nll_costs = self.nlli[-1]
        self.nll_cost = self.lam_nll[0] * T.mean(self.nll_costs)
        self.nll_bounds = self.nll_costs.ravel() + self.kld_q.ravel()
        self.nll_bound = T.mean(self.nll_bounds)
        ########################################
        # CONSTRUCT THE REST OF THE JOINT COST #
        ########################################
        param_reg_cost = self._construct_reg_costs()
        self.reg_cost = self.lam_l2w[0] * param_reg_cost
        self.joint_cost = self.nll_cost + self.kld_cost + self.reg_cost
        ##############################
        # CONSTRUCT A PER-TRIAL COST #
        ##############################
        self.obs_costs = self.nll_costs + self.kld_costs

        # Get the gradient of the joint cost for all optimizable parameters
        print("Computing gradients of self.joint_cost...")
        self.joint_grads = OrderedDict()
        grad_list = T.grad(self.joint_cost, self.joint_params)
        for i, p in enumerate(self.joint_params):
            self.joint_grads[p] = grad_list[i]

        # Construct the updates for the generator and inferencer networks
        self.joint_updates = get_adam_updates(params=self.joint_params, \
                grads=self.joint_grads, alpha=self.lr, \
                beta1=self.mom_1, beta2=self.mom_2, \
                mom2_init=1e-3, smoothing=1e-5, max_grad_norm=10.0)
        for k, v in self.scan_updates.items():
            self.joint_updates[k] = v

        # Construct a function for jointly training the generator/inferencer
        print("Compiling training function...")
        self.train_joint = self._construct_train_joint()
        print("Compiling free-energy sampler...")
        self.compute_fe_terms = self._construct_compute_fe_terms()
        print("Compiling best step cost computer...")
        self.compute_per_step_cost = self._construct_compute_per_step_cost()
        print("Compiling data-guided imputer sampler...")
        self.sample_imputer = self._construct_sample_imputer()
        # make easy access points for some interesting parameters
        #self.gen_inf_weights = self.p_zi_given_xi.shared_layers[0].W
        return

    def _from_si_to_x(self, si):
        """
        Convert the given si from s-space to x-space.
        """
        if self.use_p_x_given_si:
            x_pre_trans, _ = self.p_x_given_si.apply(si)
        else:
            x_pre_trans = si
        x_post_trans = self.obs_transform(x_pre_trans)
        return x_post_trans

    def set_sgd_params(self, lr=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rate
        new_lr = zero_ary + lr
        self.lr.set_value(to_fX(new_lr))
        # set momentums (use first and second order "momentum")
        new_mom_1 = zero_ary + mom_1
        self.mom_1.set_value(to_fX(new_mom_1))
        new_mom_2 = zero_ary + mom_2
        self.mom_2.set_value(to_fX(new_mom_2))
        return

    def set_lam_nll(self, lam_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_nll
        self.lam_nll.set_value(to_fX(new_lam))
        return

    def set_lam_kld(self, lam_kld_p=0.0, lam_kld_q=1.0, lam_kld_g=0.0, lam_kld_s=0.0):
        """
        Set the relative weight of prior KL-divergence vs. data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_kld_p
        self.lam_kld_p.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_q
        self.lam_kld_q.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_g
        self.lam_kld_g.set_value(to_fX(new_lam))
        new_lam = zero_ary + lam_kld_s
        self.lam_kld_s.set_value(to_fX(new_lam))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(to_fX(new_lam))
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
        self.train_switch.set_value(to_fX(new_val))
        return

    def _construct_zi_zmuv(self, xi, br):
        """
        Construct the necessary (symbolic) samples for computing through this
        GPSImputer for input (sybolic) matrix xi.
        """
        zi_zmuv = self.rng.normal( \
                size=(self.imp_steps, xi.shape[0]*br, self.z_dim), \
                avg=0.0, std=1.0, dtype=theano.config.floatX)
        return zi_zmuv

    def _construct_nll_costs(self, si, xo, xm):
        """
        Construct the negative log-likelihood part of free energy.
        """
        # average log-likelihood over the refinement sequence
        xh = self._from_si_to_x( si )
        xm_inv = 1.0 - xm # we will measure nll only where xm_inv is 1
        if self.x_type == 'bernoulli':
            ll_costs = log_prob_bernoulli(xo, xh, mask=xm_inv)
        else:
            ll_costs = log_prob_gaussian2(xo, xh, \
                    log_vars=self.bounded_logvar, mask=xm_inv)
        nll_costs = -ll_costs.flatten()
        return nll_costs

    def _construct_kld_s(self, s_i, s_j):
        """
        Compute KL(s_i || s_j) -- assuming bernoullish outputs
        """
        x_i = self._from_si_to_x( s_i )
        x_j = self._from_si_to_x( s_j )
        kld_s = (x_i * (T.log(x_i)  - T.log(x_j))) + \
                ((1.0 - x_i) * (T.log(1.0-x_i) - T.log(1.0-x_j)))
        sum_kld = T.sum(kld_s, axis=1)
        return sum_kld

    def _construct_kld_costs(self, p=1.0):
        """
        Construct the policy KL-divergence part of cost to minimize.
        """
        kld_pis = []
        kld_qis = []
        kld_gis = []
        kld_sis = [self._construct_kld_s(self.s0, self.s_null)]
        for i in range(self.imp_steps):
            kld_pis.append(T.sum(self.kldi_p2q[i]**p, axis=1))
            kld_qis.append(T.sum(self.kldi_q2p[i]**p, axis=1))
            kld_gis.append(T.sum(self.kldi_p2g[i]**p, axis=1))
            if i == 0:
                kld_sis.append(self._construct_kld_s(self.si[i], self.s0))
            else:
                kld_sis.append(self._construct_kld_s(self.si[i], self.si[i-1]))
        # compute the batch-wise costs
        kld_pi = sum(kld_pis)
        kld_qi = sum(kld_qis)
        kld_gi = sum(kld_gis)
        kld_si = sum(kld_sis)
        return [kld_pi, kld_qi, kld_gi, kld_si]

    def _construct_reg_costs(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network activations and parameters.
        """
        param_reg_cost = sum([T.sum(p**2.0) for p in self.joint_params])
        return param_reg_cost

    def _construct_compute_fe_terms(self):
        """
        Construct a function for computing terms in variational free energy.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        # construct values to output
        nll = self.nll_costs.flatten()
        kld = self.kld_q.flatten()
        # compile theano function for a one-sample free-energy estimate
        fe_term_sample = theano.function(inputs=[ xi, xo, xm ], \
                outputs=[nll, kld], \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.x_mask: xm, \
                        self.zi_zmuv: zizmuv}, \
                updates=self.scan_updates, \
                on_unused_input='ignore')
        # construct a wrapper function for multi-sample free-energy estimate
        def fe_term_estimator(XI, XO, XM, sample_count=20, use_guide_policy=True):
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from guide policies (i.e. variational q)
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from model's imputation policy
                self.set_train_switch(switch_val=0.0)
            # compute a multi-sample estimate of variational free-energy
            nll_sum = np.zeros((XI.shape[0],))
            kld_sum = np.zeros((XI.shape[0],))
            for i in range(sample_count):
                result = fe_term_sample(XI, XO, XM)
                nll_sum += result[0].ravel()
                kld_sum += result[1].ravel()
            mean_nll = nll_sum / float(sample_count)
            mean_kld = kld_sum / float(sample_count)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            if not use_guide_policy:
                # no KLd if samples are from the primary policy...
                mean_kld = 0.0 * mean_kld
            return [mean_nll, mean_kld]
        return fe_term_estimator

    def _construct_compute_per_step_cost(self):
        """
        Construct a theano function for computing the best possible cost
        achieved by sequential imputation.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        # construct symbolic variables for the step-wise cost
        init_nll = T.mean(self.nll0)
        init_kld = T.mean(T.sum(self.kldh_q2p, axis=1))
        step_nll = T.mean(self.nlli, axis=1).flatten()
        step_kld = T.mean(T.sum(self.kldi_q2p, axis=2), axis=1).flatten()
        # compile theano function for computing the step-wise cost
        step_cost_func = theano.function(inputs=[xi, xo, xm], \
                    outputs=[init_nll, step_nll, init_kld, step_kld], \
                    givens={ self.x_in: xi, \
                             self.x_out: xo, \
                             self.x_mask: xm, \
                             self.zi_zmuv: zizmuv }, \
                    updates=self.scan_updates, \
                    on_unused_input='ignore')
        def step_cost_computer(XI, XO, XM, sample_count=20):
            # compute a multi-sample estimate of variational free-energy
            step_nll_sum = np.zeros((1+self.imp_steps,))
            step_kld_sum = np.zeros((1+self.imp_steps,))
            for i in range(sample_count):
                result = step_cost_func(XI, XO, XM)
                step_nll_sum[0] += result[0]
                step_nll_sum[1:] += result[1].ravel()
                step_kld_sum[0] += result[2]
                step_kld_sum[1:] += result[3].ravel()
            mean_step_nll = step_nll_sum / float(sample_count)
            mean_step_kld = step_kld_sum / float(sample_count)
            return [mean_step_nll, mean_step_kld]
        return step_cost_computer

    def _construct_train_joint(self):
        """
        Construct theano function to train all networks jointly.
        """
        # setup some symbolic variables for theano to deal with
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        br = T.lscalar()
        zizmuv = self._construct_zi_zmuv(xi, br)
        # collect the outputs to return from this function
        outputs = [self.joint_cost, self.nll_bound, self.nll_cost, \
                   self.kld_cost, self.reg_cost, self.obs_costs]
        # compile the theano function
        func = theano.function(inputs=[ xi, xo, xm, br ], \
                outputs=outputs, \
                givens={ self.x_in: xi.repeat(br, axis=0), \
                         self.x_out: xo.repeat(br, axis=0), \
                         self.x_mask: xm.repeat(br, axis=0), \
                         self.zi_zmuv: zizmuv }, \
                updates=self.joint_updates, \
                on_unused_input='ignore')
        return func

    def _construct_sample_imputer(self):
        """
        Construct a function for drawing samples from the distribution
        generated by running this imputer.
        """
        xi = T.matrix()
        xo = T.matrix()
        xm = T.matrix()
        zizmuv = self._construct_zi_zmuv(xi, 1)
        oputs = [self.x_init, self._from_si_to_x(self.s0)] + \
                [self._from_si_to_x(self.si[i]) for i in range(self.imp_steps)]
        sample_func = theano.function(inputs=[xi, xo, xm], outputs=oputs, \
                givens={self.x_in: xi, \
                        self.x_out: xo, \
                        self.x_mask: xm, \
                        self.zi_zmuv: zizmuv}, \
                updates=self.scan_updates, \
                on_unused_input='ignore')
        def imputer_sampler(XI, XO, XM, use_guide_policy=False):
            XI = to_fX( XI )
            XO = to_fX( XO )
            XM = to_fX( XM )
            # set model to desired generation mode
            old_switch = self.train_switch.get_value(borrow=False)
            if use_guide_policy:
                # take samples from guide policies (i.e. variational q)
                self.set_train_switch(switch_val=1.0)
            else:
                # take samples from model's imputation policy
                self.set_train_switch(switch_val=0.0)
            # draw guided/unguided conditional samples
            model_samps = sample_func(XI, XO, XM)
            # set model back to either training or generation mode
            self.set_train_switch(switch_val=old_switch)
            # reverse engineer the "masked" samples...
            masked_samps = []
            for xs in model_samps:
                xsm = (XM * XI) + ((1.0 - XM) * xs)
                masked_samps.append(xsm)
            return model_samps, masked_samps
        return imputer_sampler

    def save_to_file(self, f_name=None):
        """
        Dump important stuff to a Python pickle, so that we can reload this
        model later.
        """
        assert(not (f_name is None))
        f_handle = file(f_name, 'wb')
        # dump the dict self.params, which just holds "simple" python values
        cPickle.dump(self.params, f_handle, protocol=-1)
        # make a copy of self.shared_param_dicts, with numpy arrays in place
        # of the theano shared variables
        numpy_param_dicts = {}
        for key in self.shared_param_dicts:
            numpy_ary = self.shared_param_dicts[key].get_value(borrow=False)
            numpy_param_dicts[key] = numpy_ary
        # dump the numpy version of self.shared_param_dicts to pickle file
        cPickle.dump(numpy_param_dicts, f_handle, protocol=-1)
        # get numpy dicts for each of the "child" models that we must save
        child_model_dicts = {}
        child_model_dicts['p_h_given_x'] = self.p_h_given_x.save_to_dict()
        child_model_dicts['p_s0_given_h'] = self.p_s0_given_h.save_to_dict()
        child_model_dicts['p_zi_given_xi'] = self.p_zi_given_xi.save_to_dict()
        child_model_dicts['p_sip1_given_zi'] = self.p_sip1_given_zi.save_to_dict()
        child_model_dicts['p_x_given_si'] = self.p_x_given_si.save_to_dict()
        child_model_dicts['q_h_given_x'] = self.q_h_given_x.save_to_dict()
        child_model_dicts['q_zi_given_xi'] = self.q_zi_given_xi.save_to_dict()
        # dump the numpy child model dicts to the pickle file
        cPickle.dump(child_model_dicts, f_handle, protocol=-1)
        f_handle.close()
        return

def load_gpsimputer_from_file(f_name=None, rng=None):
    """
    Load a clone of some previously trained model.
    """
    from InfNet import load_infnet_from_dict
    from HydraNet import load_hydranet_from_dict
    assert(not (f_name is None))
    pickle_file = open(f_name)
    # reload the basic python parameters
    self_dot_params = cPickle.load(pickle_file)
    # reload the theano shared parameters
    self_dot_numpy_param_dicts = cPickle.load(pickle_file)
    self_dot_shared_param_dicts = {}
    for key in self_dot_numpy_param_dicts:
        val = to_fX(self_dot_numpy_param_dicts[key])
        self_dot_shared_param_dicts[key] = theano.shared(val)
    # reload the child models
    child_model_dicts = cPickle.load(pickle_file)
    xd = T.matrix()
    p_h_given_x = load_infnet_from_dict( \
            child_model_dicts['p_h_given_x'], rng=rng, Xd=xd)
    p_s0_given_h = load_hydranet_from_dict( \
            child_model_dicts['p_s0_given_h'], rng=rng, Xd=xd)
    p_zi_given_xi = load_infnet_from_dict( \
            child_model_dicts['p_zi_given_xi'], rng=rng, Xd=xd)
    p_sip1_given_zi = load_hydranet_from_dict( \
            child_model_dicts['p_sip1_given_zi'], rng=rng, Xd=xd)
    p_x_given_si = load_hydranet_from_dict( \
            child_model_dicts['p_x_given_si'], rng=rng, Xd=xd)
    q_h_given_x = load_infnet_from_dict( \
            child_model_dicts['q_h_given_x'], rng=rng, Xd=xd)
    q_zi_given_xi = load_infnet_from_dict( \
            child_model_dicts['q_zi_given_xi'], rng=rng, Xd=xd)
    # now, create a new GPSImputerWI based on the loaded data
    xi = T.matrix()
    xm = T.matrix()
    xo = T.matrix()
    clone_net = GPSImputerWI(rng=rng, \
                             x_in=xi, x_mask=xm, x_out=xo, \
                             p_h_given_x=p_h_given_x, \
                             p_s0_given_h=p_s0_given_h, \
                             p_zi_given_xi=p_zi_given_xi, \
                             p_sip1_given_zi=p_sip1_given_zi, \
                             p_x_given_si=p_x_given_si, \
                             q_h_given_x=q_h_given_x, \
                             q_zi_given_xi=q_zi_given_xi, \
                             params=self_dot_params, \
                             shared_param_dicts=self_dot_shared_param_dicts)
    # helpful output
    print("==================================================")
    print("LOADED GPSImputerWI WITH PARAMS:")
    for k in self_dot_params:
        print("    {0:s}: {1:s}".format(str(k), str(self_dot_params[k])))
    print("==================================================")
    return clone_net


if __name__=="__main__":
    print("Hello world!")







##############
# EYE BUFFER #
##############
