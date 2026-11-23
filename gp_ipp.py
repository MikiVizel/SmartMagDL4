"""CAtNIPP GP belief in fixed normalized magnetic units."""
from itertools import product
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score
from magnetic_simulator import MagneticConfig
from parameters import ADAPTIVE_TH


class GaussianProcessForIPP:
    def __init__(self,node_coords,config=None):
        self.config=config or MagneticConfig.from_environment()
        self.kernel=Matern(length_scale=self.config.gp_length_scale)
        self.gp=GaussianProcessRegressor(kernel=self.kernel,optimizer=None,
            alpha=max((self.config.gp_noise_uT/self.config.scale_uT)**2,1e-10))
        self.observed_points=[]; self.observed_value=[]
        self.node_coords=node_coords

    @staticmethod
    def grid():
        return np.array(list(product(np.linspace(0,1,30),repeat=2)))

    def add_observed_point(self,point_pos,value,measurement=True):
        self.observed_points.append(np.asarray(point_pos).copy())
        self.observed_value.append(float(np.asarray(value).item()))

    def update_gp(self):
        if self.observed_points:
            self.gp.fit(np.asarray(self.observed_points).reshape(-1,2),np.asarray(self.observed_value))

    def predict(self, X, return_std=False):
        return self.gp.predict(X, return_std=return_std)

    def update_node(self):
        return self.predict(self.node_coords,return_std=True)

    def evaluate_RMSE(self,y_true):
        return float(np.sqrt(np.mean((self.predict(self.grid()).ravel()-np.asarray(y_true).ravel())**2)))

    def evaluate_F1score(self,y_true):
        # Historical interface name; this has always been R², NOT an F1 score.
        return float(r2_score(y_true,self.predict(self.grid())))

    def evaluate_cov_trace(self,X=None):
        X=self.grid() if X is None or len(X)==0 else X
        _,std=self.gp.predict(X,return_std=True)
        return float(np.sum(std**2))

    def evaluate_mutual_info(self,X=None):
        # Preserve upstream diagnostic; not the observation-noise MI formula.
        X=self.grid() if X is None or len(X)==0 else X
        _,cov=self.gp.predict(X,return_cov=True)
        return float(.5*np.linalg.slogdet(np.eye(len(X))+.01*cov)[1])

    def get_high_info_area(self,t=ADAPTIVE_TH,beta=1):
        X=self.grid(); mean,std=self.predict(X,return_std=True)
        selected=X[mean+beta*std>=t]
        return selected if len(selected) else X
