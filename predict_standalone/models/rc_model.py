import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time
import numpy.linalg as linalg
from scipy import sparse
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist, squareform


class Reservoir(object):    
    r"""
        Build a reservoir and compute the sequence of the internal states.
    """

    def __init__(self, 
                 n_internal_units=100, 
                 spectral_radius=0.99, 
                 leak=None,
                 connectivity=0.3, 
                 input_scaling=0.2, 
                 noise_level=0.0, 
                 circle=False):
       
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak
        self._input_weights = None

        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units, spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units, connectivity, spectral_radius)

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
        return internal_weights
    
    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        internal_weights = sparse.rand(n_internal_units, n_internal_units,
                                       density=connectivity).todense()
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       
        return internal_weights

    def _compute_state_matrix(self, X, n_drop=0, previous_state=None):
        N, T, _ = X.shape
        if previous_state is None:
            previous_state = np.zeros((N, self._n_internal_units), dtype=float)
        if T - n_drop > 0:
            window_size = T - n_drop
        else:
            window_size = T
        state_matrix = np.empty((N, window_size, self._n_internal_units), dtype=float)
        for t in range(T):
            current_input = X[:, t, :]
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T
            if T - n_drop > 0 and t > n_drop - 1:
                state_matrix[:, t - n_drop, :] = previous_state
            elif T - n_drop <= 0:
                state_matrix[:, t, :] = previous_state
        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True, initial_state=None):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0*np.random.binomial(1, 0.5 , [self._n_internal_units, V]) - 1.0)*self._input_scaling
        states = self._compute_state_matrix(X, n_drop, previous_state=initial_state)
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)
        return states


class RC(nn.Module):
    """RC模型 - 继承nn.Module便于集成"""
    
    def __init__(self, config):
        super(RC, self).__init__()
        self.input_channels = config['input_channels']
        self.seq_len = config['seq_len']
        self.num_classes = config['num_classes']
        self.n_internal_units = config.get('n_internal_units', 100)
        self.spectral_radius = config.get('spectral_radius', 0.99)
        self.connectivity = config.get('connectivity', 0.3)
        self.input_scaling = config.get('input_scaling', 0.2)
        self.noise_level = config.get('noise_level', 0.0)
        self.bidir = config.get('bidir', False)
        self.circle = config.get('circle', False)
        self.mts_rep = config.get('mts_rep', 'mean')
        self.w_ridge = config.get('w_ridge', 1.0)
        
        self.reservoir = Reservoir(
            n_internal_units=self.n_internal_units,
            spectral_radius=self.spectral_radius,
            leak=None,
            connectivity=self.connectivity,
            input_scaling=self.input_scaling,
            noise_level=self.noise_level,
            circle=self.circle
        )
        self.ridge = Ridge(alpha=self.w_ridge)
        self.is_fitted = False
        self._output_dim = None
    
    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        res_states = self.reservoir.get_states(X, n_drop=0, bidir=self.bidir)
        if self.mts_rep == 'mean':
            input_repr = np.mean(res_states, axis=1)
        elif self.mts_rep == 'last':
            input_repr = res_states[:, -1, :]
        else:
            input_repr = np.mean(res_states, axis=1)
        self._output_dim = input_repr.shape[1]
        self.ridge.fit(input_repr, y)
        self.is_fitted = True
    
    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("RC模型必须先调用fit()进行训练")
        x_np = x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        res_states = self.reservoir.get_states(x_np, n_drop=0, bidir=self.bidir)
        if self.mts_rep == 'mean':
            input_repr = np.mean(res_states, axis=1)
        else:
            input_repr = res_states[:, -1, :]
        logits = self.ridge.predict(input_repr)
        logits = torch.from_numpy(logits).float()
        return logits
    
    def save_checkpoint(self, path):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': {
                'ridge_model': pickle.dumps(self.ridge),
                'is_fitted': self.is_fitted,
                'output_dim': self._output_dim,
            },
            'config': {
                'input_channels': self.input_channels,
                'seq_len': self.seq_len,
                'num_classes': self.num_classes,
                'n_internal_units': self.n_internal_units,
                'spectral_radius': self.spectral_radius,
                'connectivity': self.connectivity,
                'input_scaling': self.input_scaling,
                'noise_level': self.noise_level,
                'bidir': self.bidir,
                'circle': self.circle,
                'mts_rep': self.mts_rep,
                'w_ridge': self.w_ridge,
            }
        }
        torch.save(checkpoint, path)
        print(f"✓ RC模型检查点已保存: {path}")
    
    def load_checkpoint(self, path):
        import pickle
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint
        try:
            if isinstance(model_state, dict) and 'ridge_model' in model_state:
                self.ridge = pickle.loads(model_state['ridge_model'])
                self.is_fitted = model_state.get('is_fitted', True)
                self._output_dim = model_state.get('output_dim', None)
            else:
                self.ridge = pickle.loads(model_state.get('ridge_model', model_state))
                self.is_fitted = True
                self._output_dim = None
        except Exception as e:
            raise RuntimeError(f"无法加载RC模型检查点: {e}\n检查点内容: {model_state.keys() if isinstance(model_state, dict) else type(model_state)}")
        print(f"✓ RC模型检查点已加载: {path}")
        print(f"  - 模型是否已训练: {self.is_fitted}")
        print(f"  - 输出维度: {self._output_dim}")
    
    def state_dict(self):
        import pickle
        return {
            'ridge_model': pickle.dumps(self.ridge),
            'is_fitted': self.is_fitted,
            'output_dim': self._output_dim,
        }
    
    def load_state_dict(self, state_dict, strict=True):
        import pickle
        self.ridge = pickle.loads(state_dict['ridge_model'])
        self.is_fitted = state_dict['is_fitted']
        self._output_dim = state_dict['output_dim']
        return self
