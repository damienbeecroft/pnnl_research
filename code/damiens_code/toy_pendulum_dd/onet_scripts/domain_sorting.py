from anytree import NodeMixin

class DomainTree:
    """
    Author:         Damien Beecroft
    Domain_Tree:    A super of the Domain class that stores important functions for working with
                    the domain decomposition tree.
    """
    def __init__(self):
        self.dnn_size = [3,80,80,80,2]
        self.nonlin_size = [3,80,80,80,2]
        self.lin_size = [3,4,2]
    
    def find_interior_pts(self): # find which of the parent's points are in the current domain
        parent_pts = self.parent.pts # get the parent's points
        verts = self.vertices
        mask = [((verts[0] <= pt) & (pt <= verts[1])).all() for pt in parent_pts] # find which of the parent's points are in the current domain
        domain_pts = parent_pts[mask]
        return domain_pts

class Domain(DomainTree,NodeMixin):
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
    net_sizes:  [base net shape, nonlinear net shape, linear net shape]. The base net is the 
    """
    def __init__(self,vertices,parent=None,children=None,pts=None):
        super(Domain,self).__init__()
        self.vertices = vertices # two opposite vertices that define the n-dimensional box
        self.parent = parent # parent domain of the current domain

        if parent: # set level
            self.lvl = parent.lvl + 1
            self.pts = self.find_interior_pts() # otherwise, find which points are in the domain of the current domain
            # self.dnn = 
        else:
            self.lvl = 0
            self.pts = pts # if this is the root node, just assign points
            # self.mf_net = 

        if children: # set children
            self.children = children