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
        
        Parameters:
        ------------
        n_internal_units : int (default ``100``)
            Processing units in the reservoir.
        spectral_radius : float (default ``0.99``)
            Largest eigenvalue of the reservoir matrix of connection weights.
            To ensure the Echo State Property, set ``spectral_radius <= leak <= 1``)
        leak : float (default ``None``)
            Amount of leakage in the reservoir state update. 
            If ``None`` or ``1.0``, no leakage is used.
        connectivity : float (default ``0.3``)
            Percentage of nonzero connection weights.
            Unused in circle reservoir.
        input_scaling : float (default ``0.2``)
            Scaling of the input connection weights.
            Note that the input weights are randomly drawn from ``{-1,1}``.
        noise_level : float (default ``0.0``)
            Standard deviation of the Gaussian noise injected in the state update.
        circle : bool (default ``False``)
            Generate determinisitc reservoir with circle topology where each connection 
            has the same weight.
        """

    def __init__(self, 
                 n_internal_units=100, 
                 spectral_radius=0.99, 
                 leak=None,
                 connectivity=0.3, 
                 input_scaling=0.2, 
                 noise_level=0.0, 
                 circle=False):
       
        # Initialize hyperparameters
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                    n_internal_units,
                    spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius)


    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        """Generate internal weights with circular topology.
        """
        
        # Construct reservoir with circular topology
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = 1.0
        for i in range(n_internal_units-1):
            internal_weights[i+1,i] = 1.0
            
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius 
                
        return internal_weights
    
    
    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):
        """Generate internal weights with a sparse, uniformly random topology.
        """

        # Generate sparse, uniformly distributed weights.
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        
        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max)/spectral_radius       

        return internal_weights


    def _compute_state_matrix(self, X, n_drop=0, previous_state=None):
        """Compute the reservoir states on input data X.
        """

        N, T, _ = X.shape
        if previous_state is None:
            previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        if T - n_drop > 0:
            window_size = T - n_drop
        else:
            window_size = T
        state_matrix = np.empty((N, window_size, self._n_internal_units), dtype=float)

        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N)*self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if T - n_drop > 0 and t > n_drop - 1:
                state_matrix[:, t - n_drop, :] = previous_state
            elif T - n_drop <= 0:
                state_matrix[:, t, :] = previous_state

        return state_matrix


    def get_states(self, X, n_drop=0, bidir=True, initial_state=None):
        r"""
        Compute reservoir states and return them.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each
            time point.
        n_drop : int (default is ``0``)
            Washout period, i.e., number of initial samples to drop due to the transient phase.
        bidir : bool (default is ``True``)
            If ``True``, use bidirectional reservoir
        initial_state : np.ndarray (default is ``None``)
            Initialize the first state of the reservoir to the given value.
            If ``None``, the initial states is a zero-vector. 

        Returns:
        ------------
        states : np.ndarray
            Reservoir states, 3D array of shape ``[N,T,n_internal_units]``, where ``N`` is the number
            of time series, ``T`` is the length of each time series, and ``n_internal_units`` is the
            number of processing units in the reservoir.
        """

        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0*np.random.binomial(1, 0.5 , [self._n_internal_units, V]) - 1.0)*self._input_scaling

        # Compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop, previous_state=initial_state)
    
        # Reservoir states on time reversed input
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
        
        # RC参数
        self.n_internal_units = config.get('n_internal_units', 100)
        self.spectral_radius = config.get('spectral_radius', 0.99)
        self.connectivity = config.get('connectivity', 0.3)
        self.input_scaling = config.get('input_scaling', 0.2)
        self.noise_level = config.get('noise_level', 0.0)
        self.bidir = config.get('bidir', False)
        self.circle = config.get('circle', False)
        self.mts_rep = config.get('mts_rep', 'mean')
        self.w_ridge = config.get('w_ridge', 1.0)
        
        # 初始化Reservoir
        self.reservoir = Reservoir(
            n_internal_units=self.n_internal_units,
            spectral_radius=self.spectral_radius,
            leak=None,
            connectivity=self.connectivity,
            input_scaling=self.input_scaling,
            noise_level=self.noise_level,
            circle=self.circle
        )
        
        # Ridge分类器
        self.ridge = Ridge(alpha=self.w_ridge)
        self.is_fitted = False
        
        # 输出层
        self._output_dim = None
    
    def fit(self, X, y):
        """
        训练RC模型
        
        Args:
            X: [batch_size, seq_len, input_channels]
            y: [batch_size] 或 [batch_size, num_classes]
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # 如果y是one-hot编码，转换为类别标签
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        
        # 获取reservoir状态
        res_states = self.reservoir.get_states(X, n_drop=0, bidir=self.bidir)
        
        # 提取表示 (mean representation)
        if self.mts_rep == 'mean':
            input_repr = np.mean(res_states, axis=1)
        elif self.mts_rep == 'last':
            input_repr = res_states[:, -1, :]
        else:
            input_repr = np.mean(res_states, axis=1)
        
        self._output_dim = input_repr.shape[1]
        
        # 拟合ridge分类器
        self.ridge.fit(input_repr, y)
        self.is_fitted = True
    
    def forward(self, x):
        """
        推理
        
        Args:
            x: [batch_size, seq_len, input_channels]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        if not self.is_fitted:
            raise RuntimeError("RC模型必须先调用fit()进行训练")
        
        x_np = x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x
        
        # 获取reservoir状态
        res_states = self.reservoir.get_states(x_np, n_drop=0, bidir=self.bidir)
        
        # 提取表示
        if self.mts_rep == 'mean':
            input_repr = np.mean(res_states, axis=1)
        else:
            input_repr = res_states[:, -1, :]
        
        # 预测
        logits = self.ridge.predict(input_repr)
        
        # 转换为tensor
        logits = torch.from_numpy(logits).float()
        
        return logits
    
    def save_checkpoint(self, path):
        """
        保存检查点，包括已拟合的ridge模型和reservoir配置
        
        Args:
            path: 保存路径
        """
        import pickle
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 统一使用标准 PyTorch 检查点格式
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
        """
        加载检查点，恢复已拟合的ridge模型
        支持多种检查点格式以保证兼容性
        
        Args:
            path: 检查点路径
        """
        import pickle
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # 处理多种检查点格式
        # 格式1: {'model_state_dict': {...}}
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        # 格式2: {'model_state': {...}} (旧格式)
        elif 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        # 格式3: 直接是状态字典
        else:
            model_state = checkpoint
        
        # 尝试从状态字典中恢复模型
        try:
            if isinstance(model_state, dict) and 'ridge_model' in model_state:
                # 标准格式
                self.ridge = pickle.loads(model_state['ridge_model'])
                self.is_fitted = model_state.get('is_fitted', True)
                self._output_dim = model_state.get('output_dim', None)
            else:
                # 尝试直接使用为状态字典
                self.ridge = pickle.loads(model_state.get('ridge_model', model_state))
                self.is_fitted = True
                self._output_dim = None
        except Exception as e:
            raise RuntimeError(f"无法加载RC模型检查点: {e}\n检查点内容: {model_state.keys() if isinstance(model_state, dict) else type(model_state)}")
        
        print(f"✓ RC模型检查点已加载: {path}")
        print(f"  - 模型是否已训练: {self.is_fitted}")
        print(f"  - 输出维度: {self._output_dim}")
    
    def state_dict(self):
        """
        返回模型状态字典(PyTorch标准接口)
        RC模型需要特殊处理因为使用了sklearn模型
        """
        import pickle
        return {
            'ridge_model': pickle.dumps(self.ridge),
            'is_fitted': self.is_fitted,
            'output_dim': self._output_dim,
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """
        加载模型状态字典
        """
        import pickle
        self.ridge = pickle.loads(state_dict['ridge_model'])
        self.is_fitted = state_dict['is_fitted']
        self._output_dim = state_dict['output_dim']
        return self




class tensorPCA:
    r"""
    Compute PCA on a dataset of multivariate time series represented as a 3-dimensional tensor
    and reduce the size along the third dimension from ``[N, T, V]`` to ``[N, T, D]``, where ``D <= V`` .

    The input dataset must be a 3-dimensional tensor, where the first dimension ``N`` represents 
    the number of observations, the second dimension ``T`` represents the number of time steps 
    in the time series, and the third dimension ``V`` represents the number of variables in the time series.

    Parameters
    ----------
    n_components : int
        The number of principal components to keep after the dimensionality reduction. This
        determines the size of the third dimension ``D`` in the output tensor.
    """

    def __init__(self, n_components):
        self.n_components=n_components
        self.first_eigs = None
        
    def fit(self, X):
        r"""
        Fit the tensorPCA model to the input dataset ``X``.
        
        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        None
        """
        if len(X.shape) != 3:
            raise RuntimeError('Input must be a 3d tensor')
        
        Xt = np.swapaxes(X,1,2)  # [N,T,V] --> [N,V,T]
        Xm = np.expand_dims(np.mean(X, axis=0), axis=0) # mean sample
        Xmt = np.swapaxes(Xm,1,2)
        
        C = np.tensordot(X-Xm,Xt-Xmt,axes=([1,0],[2,0])) / (X.shape[0]-1) # covariance of 0-mode slices
        
        # Sort eigenvalues of covariance matrix
        eigenValues, eigenVectors = linalg.eig(C)
        idx = eigenValues.argsort()[::-1]   
        eigenVectors = eigenVectors[:,idx]
        
        self.first_eigs = eigenVectors[:,:self.n_components]
        
    def transform(self, X):
        r"""
        Transform the input dataset X using the tensorPCA model.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        Xpca : np.ndarray
            Transformed time series, 3D array of shape ``[N,T,D]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``D`` is the number of principal components.
        """
        return np.einsum('klj,ji->kli',X,self.first_eigs)
    
    def fit_transform(self, X):
        r"""
        Fit the tensorPCA model to the input dataset ``X`` and transform it.

        Parameters:
        ------------
        X : np.ndarray
            Time series, 3D array of shape ``[N,T,V]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``V`` is the number of variables in each.

        Returns:
        ------------
        Xpca : np.ndarray
            Transformed time series, 3D array of shape ``[N,T,D]``, where ``N`` is the number of time series,
            ``T`` is the length of each time series, and ``D`` is the number of principal components.
        """
        self.fit(X)
        return self.transform(X)

