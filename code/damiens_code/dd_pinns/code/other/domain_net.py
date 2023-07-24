from anytree import NodeMixin
from utils_fs_v2 import DNN
from jax import random
from jax.example_libraries import optimizers

class DomainNetSuper:
    """
    Author:         Damien Beecroft
    Domain_Tree:    A super of the Domain class that stores important functions for working with
                    the domain decomposition tree.
    """
    def __init__(self):
        # self.dnn_size = [3,80,80,80,2]
        # self.nonlin_size = [3,80,80,80,2]
        # self.lin_size = [3,4,2]
        pass
    
    def find_interior_pts(self): # find which of the parent's points are in the current domain
        parent_pts = self.parent.pts # get the parent's points
        verts = self.vertices
        mask = [((verts[0] <= pt) & (pt <= verts[1])).all() for pt in parent_pts] # find which of the parent's points are in the current domain
        domain_pts = parent_pts[mask]
        return domain_pts

class DomainNet(DomainNetSuper,NodeMixin):
    """
    Author:     Damien Beecroft
    Domain:     A class that defines properties of the relevant domains used for tracking where each
                neural network has support. The local properties of the nodes are defined here. The "global"
                variables are stored in the support class: DomainTree.
    =================================================================================================
    vertices:   Two opposite points on the hyperrectangle used to define the domain. 
    parent:     The node corresponding to the domain that the current node's domain is the immediate subset of
    children:   The nodes whose domains are the immediate subsets of the current domain.
    pts:        List of collocation points for the neural network to be evaluated at. This value is only passed 
                in for the root node. All other nodes determine the points determine their interior points through
                the find_interior points function in the Domain_Tree class.
                NEED TO COMMENT ON OTHER VARIABLES 
    """
    def __init__(self,vertices,layers_branch_low, ics_weight, res_weight, data_weight,
                 params_prev, lr,parent=None,children=None,pts=None):
        #===========================================================================================
        # My Initialization Variables
        #===========================================================================================
        super(DomainNet,self).__init__()
        self.vertices = vertices # two opposite vertices that define the n-dimensional box
        self.parent = parent # parent domain of the current domain

        if parent: # set level
            self.lvl = parent.lvl + 1
            self.pts = self.find_interior_pts() # otherwise, find which points are in the domain of the current domain
        else:
            self.lvl = 0
            self.pts = pts # if this is the root node, just assign points

        if children: # set children
            self.children = children

        #===========================================================================================
        # Amanda's Initialization Variables
        #===========================================================================================
        self.init_low, self.apply_low = DNN(layers_branch_low)
        params_low = self.init_low(random.PRNGKey(1))
        params = (params_low)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        self.itercount = itertools.count()

        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.data_weight = data_weight


        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_data_log = []